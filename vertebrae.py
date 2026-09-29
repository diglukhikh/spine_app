import cv2
import numpy as np
import json
import os
from scipy.signal import find_peaks


# ================= утилиты =================

def load_polygon(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        if shape.get("label") == "spine":
            pts = np.array(shape["points"], dtype=np.float32)
            return pts.astype(np.int32)
    raise ValueError("no 'spine' polygon in JSON")


def polygon_mask(shape_hw, polygon: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def spine_row_bounds(spine_mask: np.ndarray):
    """
    Для каждой строки — левая и правая границы полигона.
    Возвращает (left[y], right[y], cx[y], valid[y]).
    """
    h, w = spine_mask.shape
    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    cx = np.full(h, np.nan, dtype=np.float32)
    valid = np.zeros(h, dtype=bool)

    for y in range(h):
        row = spine_mask[y]
        idx = np.where(row > 0)[0]
        if idx.size < 4:
            continue
        left[y] = idx.min()
        right[y] = idx.max() + 1
        cx[y] = (left[y] + right[y]) / 2.0
        valid[y] = True

    return left, right, cx, valid


def interpolate_nan(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.sum() == 0:
        return arr
    out = arr.copy()
    idx = np.arange(len(arr))
    out[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return out


def build_axis_profile(gray: np.ndarray,
                       left: np.ndarray,
                       right: np.ndarray,
                       valid: np.ndarray,
                       smooth_win: int = 9):
    """
    Строит одномерный профиль яркости ВДОЛЬ позвоночника.
    Для каждой строки y: средняя яркость по X от left[y] до right[y].
    Учитывает сколиоз — границы берутся из полигона, а не вертикальные.
    """
    h = gray.shape[0]
    profile = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        if not valid[y]:
            continue
        l = int(max(0, left[y]))
        r = int(min(gray.shape[1], right[y]))
        if r - l < 4:
            continue
        # берём среднюю яркость поперёк позвонка
        profile[y] = gray[y, l:r].mean()

    profile = interpolate_nan(profile, valid)

    # сглаживание профиля — чтобы убрать шум, но сохранить пики позвонков
    if smooth_win % 2 == 0:
        smooth_win += 1
    if smooth_win >= 3:
        kernel = np.ones(smooth_win, dtype=np.float32) / smooth_win
        profile_s = np.convolve(profile, kernel, mode='same')
    else:
        profile_s = profile

    return profile_s


def subtract_trend(profile: np.ndarray, trend_win: int = 51) -> np.ndarray:
    """
    Убирает медленный тренд (общая яркость позвоночника меняется по высоте).
    Остаются локальные пики — тела позвонков.
    """
    if trend_win % 2 == 0:
        trend_win += 1
    kernel = np.ones(trend_win, dtype=np.float32) / trend_win
    trend = np.convolve(profile, kernel, mode='same')
    return profile - trend


def find_vertebrae_peaks(profile_detrended: np.ndarray,
                         valid: np.ndarray,
                         min_period: int = 15,
                         max_period: int = 90,
                         peak_height_frac: float = 0.3,
                         prominence_frac: float = 0.3):
    """
    Ищет позвонки как пики на детрендированном профиле.
    Возвращает индексы Y — центры позвонков.
    """
    # работаем только по валидному диапазону
    ys = np.where(valid)[0]
    if ys.size < 20:
        return np.array([], dtype=int)

    y0, y1 = int(ys.min()), int(ys.max())
    seg = profile_detrended[y0:y1+1]

    # порог по высоте — доля от std и максимума
    pos = seg[seg > 0]
    if pos.size == 0:
        return np.array([], dtype=int)

    height_thr = max(pos.std() * peak_height_frac, pos.max() * peak_height_frac * 0.5)
    prominence_thr = pos.std() * prominence_frac

    peaks, props = find_peaks(
        seg,
        distance=min_period,
        height=height_thr,
        prominence=prominence_thr,
    )

    # ограничиваем макс. расстояние между пиками — если больше, значит пропустили позвонок
    if peaks.size == 0:
        return np.array([], dtype=int)

    # фильтр по периоду
    peaks_global = peaks + y0
    filtered = [peaks_global[0]]
    for p in peaks_global[1:]:
        if p - filtered[-1] <= max_period:
            filtered.append(p)
        else:
            # слишком далеко — вставим промежуточный
            n_missing = max(1, round((p - filtered[-1]) / max_period))
            step = (p - filtered[-1]) / (n_missing + 1)
            for k in range(1, n_missing + 1):
                filtered.append(int(filtered[-1] + step))
            filtered.append(p)

    return np.array(sorted(set(filtered)), dtype=int)


def vertebrae_from_peaks(peaks: np.ndarray,
                         left: np.ndarray,
                         right: np.ndarray,
                         h: int):
    """
    Для каждого пика строит bbox позвонка: ширина = от left[y] до right[y]
    (т.е. почти вся ширина позвоночника в этой строке).
    Высота = расстояние до соседних пиков / 2 с каждой стороны.
    """
    if peaks.size == 0:
        return []

    blobs = []
    for i, cy in enumerate(peaks):
        y_top = int((peaks[i-1] + cy) / 2) if i > 0 else 0
        y_bot = int((peaks[i+1] + cy) / 2) if i < len(peaks)-1 else h - 1

        # берём границы полигона на уровне центра позвонка
        l = int(left[cy])
        r = int(right[cy])

        blobs.append({
            "id": i + 1,
            "cy": int(cy),
            "cx": int((l + r) / 2),
            "x": l,
            "y": y_top,
            "w": r - l,
            "h": y_bot - y_top,
        })
    return blobs


# ================= основная функция =================

def analyze_spine(
    image_path: str,
    json_path: str,

    smooth_win: int = 17,           # окно сглаживания профиля по Y (нечётное);
                                    #   больше = плавнее профиль, шумовые всплески уходят;
                                    #   меньше = резче пики, но больше ложных срабатываний;
                                    #   ориентир: ≈ 1/2 высоты позвонка в пикселях

    trend_win: int = 36,            # окно вычитания медленного тренда (нечётное);
                                    #   больше = сильнее давит медленные изменения яркости,
                                    #   меньше = тренд слабо убирается, остаются горбы;
                                    #   ориентир: 1.5–2 × высота позвонка в пикселях

    min_period: int = 20,           # мин. расстояние между соседними пиками (пиксели);
                                    #   больше = запрещает слишком близкие позвонки,
                                    #   меньше = можно поймать два пика на одном позвонке;
                                    #   ориентир: ≈ 1/2 высоты позвонка

    max_period: int = 80,           # макс. расстояние между соседними пиками (пиксели);
                                    #   больше = разрешает «пропуски» и достраивает позвонки,
                                    #   меньше = жёстко требует равномерности;
                                    #   ориентир: ≈ 1.5 высоты позвонка

    peak_height_frac: float = 0.38, # мин. высота пика как доля от std положительных значений профиля;
                                    #   больше = отсекает слабые пики (меньше ложных),
                                    #   меньше = ловит даже слабые позвонки (больше ложных)

    prominence_frac: float = 0.38,  # мин. «выраженность» пика (насколько он выделяется
                                    #   относительно соседних впадин), доля от std;
                                    #   больше = только явные пики,
                                    #   меньше = ловит слабовыраженные позвонки

    debug: bool = False,
) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape

    polygon = load_polygon(json_path)
    spine_mask = polygon_mask((h, w), polygon)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # 1. Границы полигона по строкам
    left, right, cx, valid = spine_row_bounds(spine_mask)

    # 2. Профиль яркости вдоль оси позвоночника
    profile = build_axis_profile(img_eq, left, right, valid, smooth_win=smooth_win)

    # 3. Убираем тренд — остаются пики позвонков
    profile_detrended = subtract_trend(profile, trend_win=trend_win)

    # 4. Ищем пики
    peaks = find_vertebrae_peaks(
        profile_detrended, valid,
        min_period=min_period,
        max_period=max_period,
        peak_height_frac=peak_height_frac,
        prominence_frac=prominence_frac,
    )

    # 5. Строим bbox'ы позвонков
    blobs = vertebrae_from_peaks(peaks, left, right, h)

    result = {
        "n_vertebrae": len(blobs),
        "vertebrae": blobs,
    }

    if debug:
        # overlay
        overlay = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
        cv2.polylines(overlay, [polygon], True, (0, 200, 0), 2)

        for b in blobs:
            cv2.rectangle(overlay,
                          (b["x"], b["y"]),
                          (b["x"] + b["w"], b["y"] + b["h"]),
                          (0, 255, 255), 1)
            cv2.circle(overlay, (b["cx"], b["cy"]), 3, (0, 0, 255), -1)
            cv2.putText(overlay, str(b["id"]),
                        (b["x"], max(12, b["y"] + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

        # график профиля — визуализация
        plot_h = 300
        plot = np.zeros((plot_h, w, 3), dtype=np.uint8)
        # профиль
        pmin, pmax = profile_detrended.min(), profile_detrended.max()
        if pmax - pmin < 1e-6:
            pmax = pmin + 1
        for y in range(plot_h):
            orig_y = int(y * h / plot_h)
            val = profile_detrended[orig_y]
            x = int((val - pmin) / (pmax - pmin) * (w - 1))
            x = np.clip(x, 0, w - 1)
            cv2.circle(plot, (x, y), 1, (0, 255, 0), -1)
        # нулевая линия
        zero_x = int((0 - pmin) / (pmax - pmin) * (w - 1))
        cv2.line(plot, (zero_x, 0), (zero_x, plot_h), (100, 100, 100), 1)
        # пики
        for p in peaks:
            yy = int(p * plot_h / h)
            cv2.line(plot, (0, yy), (w, yy), (0, 0, 255), 1)
            cv2.circle(plot, (int((profile_detrended[p] - pmin) / (pmax - pmin) * (w - 1)),
                              yy), 4, (0, 0, 255), -1)

        # световая маска — покажем профиль как яркость по строкам
        light = np.zeros((h, w), dtype=np.uint8)
        pmin2, pmax2 = profile_detrended.min(), profile_detrended.max()
        if pmax2 - pmin2 < 1e-6:
            pmax2 = pmin2 + 1
        for y in range(h):
            v = (profile_detrended[y] - pmin2) / (pmax2 - pmin2)
            light[y, :] = int(np.clip(v * 255, 0, 255))
        light[spine_mask == 0] = 0
        light_vis = cv2.applyColorMap(light, cv2.COLORMAP_JET)

        result["debug_images"] = {
            "heatmap": light_vis,     # температурная карта профиля
            "profile_plot": plot,     # график профиля и пики
            "overlay": overlay,       # исходник + номера позвонков
        }

    return result


# ================= CLI =================

if __name__ == "__main__":
    import os

    # ===== НАСТРОЙКИ ЗАПУСКА ИЗ PYCHARM =====
    image_paths = [
        r"upload_dicom/upload_20260927_222413/CR000001.jpg",   # <-- поменяйте на свои пути
        # r"C:\path\to\your\image2.jpg",
    ]

    debug = True                       # True — сохранит отладочные изображения
    out_dir = "vertebrae_debug"
    # ========================================

    if debug:
        os.makedirs(out_dir, exist_ok=True)

    for img_path in image_paths:
        base = os.path.splitext(img_path)[0]
        json_path = base + ".json"
        if not os.path.exists(json_path):
            print(f"{img_path}: ERROR no JSON at {json_path}")
            continue

        try:
            r = analyze_spine(img_path, json_path, debug=debug)
        except Exception as e:
            print(f"{img_path}: ERROR {type(e).__name__}: {e}")
            continue

        print(f"{img_path}: n_vertebrae = {r['n_vertebrae']}")
        for b in r["vertebrae"]:
            print(f"  #{b['id']}: center=({b['cx']},{b['cy']}) "
                  f"bbox=({b['x']},{b['y']},{b['w']}x{b['h']})")

        if debug and "debug_images" in r:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            for name, im in r["debug_images"].items():
                cv2.imwrite(os.path.join(out_dir, f"{stem}_{name}.jpg"), im)
            print(f"  -> saved debug to {out_dir}/{stem}_*.jpg")