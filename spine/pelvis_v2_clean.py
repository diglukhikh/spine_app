import cv2
import numpy as np
import os


# Pelvis detector v2
#
#
# v2 специально сделан устойчивее к слабым костям. Поэтому
# слабая сторона не должна обязательно пройти "сильный" порог.


def _percentile_positive(a: np.ndarray, percentile: float, fallback: float = 0.0) -> float:
    """Перцентиль только положительных значений."""
    pos = a[a > 0]
    if pos.size == 0:
        return fallback
    return float(np.percentile(pos, percentile))


def _safe_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная корреляция двух одинаковых по размеру массивов."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()

    a -= a.mean()
    b -= b.mean()

    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-6:
        return 0.0

    return float(np.dot(a, b) / den)


def find_spine_columns_v2(
    bright_local: np.ndarray,
    search_frac: float = 0.55,
    peak_ratio: float = 0.30,
    pad_frac: float = 0.015,
) -> tuple:
    """
    Примерная полоса позвоночника.

    В отличие от старой версии, это только вспомогательная
    информация. Финальное решение от точности spine_x0/x1
    напрямую не зависит.
    """
    bh, bw = bright_local.shape
    col = bright_local.mean(axis=0)

    cx = bw // 2
    half_search = int(bw * search_frac / 2)

    x0 = max(0, cx - half_search)
    x1 = min(bw, cx + half_search)

    window = col[x0:x1]

    if window.size == 0 or window.max() <= 0:
        half = max(5, int(bw * 0.10 / 2))
        return max(0, cx - half), min(bw, cx + half)

    # центральную область вокруг максимума.
    peak_idx = int(np.argmax(window))
    peak = float(window[peak_idx])

    if peak <= 0:
        half = max(5, int(bw * 0.10 / 2))
        return max(0, cx - half), min(bw, cx + half)

    thr = peak * peak_ratio

    left = peak_idx
    right = peak_idx

    while left > 0 and window[left - 1] >= thr:
        left -= 1

    while right + 1 < window.size and window[right + 1] >= thr:
        right += 1

    sx0 = x0 + left
    sx1 = x0 + right + 1

    pad = max(3, int(bw * pad_frac))

    sx0 = max(0, sx0 - pad)
    sx1 = min(bw, sx1 + pad)

    return sx0, sx1


def make_local_contrast(
    bottom: np.ndarray,
    sigma: float = 9.0,
) -> np.ndarray:
    """
    Локальный контраст.

    Используется Gaussian background вместо большого box blur:
    это меньше зависит от размера костной структуры.
    """
    f = bottom.astype(np.float32)

    background = cv2.GaussianBlur(
        f,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return f - background


def make_candidate_mask(
    side_contrast: np.ndarray,
    threshold_percentile: float = 55.0,
    min_abs_contrast: float = 2.0,
) -> tuple:
    """
    Маска костных кандидатов.

    Порог вычисляется отдельно для каждой стороны.
    Это важно: левая и правая стороны могут иметь разную
    экспозицию/контраст.

    Возвращает mask, threshold.
    """
    thr = _percentile_positive(side_contrast, threshold_percentile)

    # В очень тёмных исследованиях percentile может оказаться
    # около нуля. Тогда не разрешаем маске превратиться
    # практически во весь ROI.
    thr = max(thr, min_abs_contrast)

    mask = (side_contrast > thr).astype(np.uint8)

    # Убираем единичный шум.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    # Соединяем фрагменты одной костной структуры.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )

    return mask, float(thr)


def component_score(
    area: int,
    width: int,
    height: int,
    fill_ratio: float,
    mean_contrast: float,
    cy: float,
    roi_h: int,
    roi_w: int,
) -> float:
    """
    Оценка одного кандидата.

    Здесь намеренно нет одного жёсткого критерия.
    Большая связная область получает больший вес,
    но слабая кость может компенсировать небольшой
    контраст площадью/формой/положением.
    """

    # Площадь.
    area_s = np.clip(
        area / (0.012 * roi_h * roi_w + 1e-6),
        0.0,
        1.0,
    )

    # Ширина важна: случайные тонкие линии и мелкие
    # наложения обычно проигрывают настоящей костной области.
    width_s = np.clip(
        width / (0.18 * roi_w + 1e-6),
        0.0,
        1.0,
    )

    # Высота.
    height_s = np.clip(
        height / (0.30 * roi_h + 1e-6),
        0.0,
        1.0,
    )

    # совершенно тонкой линии.
    fill_s = np.clip(fill_ratio / 0.55, 0.0, 1.0)

    # Средний локальный контраст.
    contrast_s = np.clip(mean_contrast / 25.0, 0.0, 1.0)

    # Кандидат должен находиться ближе к нижней части ROI.
    y_s = np.clip(
        (cy / (0.80 * roi_h + 1e-6)),
        0.0,
        1.0,
    )

    score = (
        0.30 * area_s +
        0.20 * width_s +
        0.10 * height_s +
        0.10 * fill_s +
        0.15 * contrast_s +
        0.15 * y_s
    )

    return float(score)


def find_best_component(
    side_mask: np.ndarray,
    side_contrast: np.ndarray,
    x_offset: int,
    roi_h: int,
    roi_w: int,
    min_area_frac: float = 0.0015,
    min_width_frac: float = 0.025,
    min_height_frac: float = 0.025,
) -> dict | None:
    """
    Ищет лучший крупный кандидат на одной стороне.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        side_mask,
        connectivity=8,
    )

    min_area = max(20, int(side_mask.size * min_area_frac))
    min_width = max(7, int(roi_w * min_width_frac))
    min_height = max(6, int(roi_h * min_height_frac))

    best = None

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < min_area:
            continue
        if w < min_width or h < min_height:
            continue

        comp = labels == i
        vals = side_contrast[comp]

        if vals.size == 0:
            continue

        mean_contrast = float(vals.mean())
        fill_ratio = float(area / (w * h + 1e-6))
        cx, cy = centroids[i]

        score = component_score(
            area=area,
            width=w,
            height=h,
            fill_ratio=fill_ratio,
            mean_contrast=mean_contrast,
            cy=float(cy),
            roi_h=roi_h,
            roi_w=roi_w,
        )

        candidate = {
            "score": score,
            "area": area,
            "x": x + x_offset,
            "y": y,
            "w": w,
            "h": h,
            "cx": float(cx + x_offset),
            "cy": float(cy),
            "mean_contrast": mean_contrast,
            "fill_ratio": fill_ratio,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def lateral_zone_evidence(
    bottom: np.ndarray,
    x0: int,
    x1: int,
    lower_start_frac: float = 0.55,
    upper_end_frac: float = 0.55,
) -> dict:
    """
    Дополнительный признак.

    Сравнивает нижнюю латеральную область с тканями выше неё.
    Настоящая тазовая кость должна давать заметное увеличение
    яркости/контраста внизу.

    Этот признак нужен именно для случаев, когда кость есть,
    но connected component получается слабым или раздробленным.
    """
    H, _ = bottom.shape

    lower_y0 = int(H * lower_start_frac)
    upper_y1 = int(H * upper_end_frac)
    upper_y0 = int(H * 0.15)

    low = bottom[lower_y0:H, x0:x1].astype(np.float32)
    upper = bottom[upper_y0:upper_y1, x0:x1].astype(np.float32)

    if low.size == 0 or upper.size == 0:
        return {
            "ratio": 0.0,
            "high_frac": 0.0,
            "score": 0.0,
        }

    low_mean = float(low.mean())
    upper_mean = float(upper.mean())

    # на очень тёмных изображениях upper_mean может быть почти 0.
    ratio = low_mean / (upper_mean + 5.0)

    # 1.0 -> почти нет отличия.
    # 2.5+ -> сильное увеличение.
    ratio_score = float(np.clip((ratio - 1.0) / 1.5, 0.0, 1.0))

    q90 = float(np.percentile(
        np.concatenate([low.ravel(), upper.ravel()]),
        90,
    ))

    high_frac = float((low > q90).mean())

    # В нормальном ROI около 10% пикселей уже будут выше q90.
    # Поэтому учитываем только заметное превышение.
    high_score = float(np.clip(
        (high_frac - 0.08) / 0.10,
        0.0,
        1.0,
    ))

    score = 0.55 * ratio_score + 0.45 * high_score

    return {
        "ratio": ratio,
        "high_frac": high_frac,
        "score": float(score),
    }


def mirror_symmetry_score(
    bottom: np.ndarray,
    center_x: float,
    left_x0: int,
    right_x1: int,
    y_start_frac: float = 0.55,
) -> float:
    """
    Мягкая проверка симметрии нижних латеральных областей.

    Это НЕ обязательное условие: таз может быть повернут/асимметричен.
    Поэтому score только слегка влияет на итог.
    """
    H, W = bottom.shape

    y0 = int(H * y_start_frac)

    # Берём одинаковую ширину слева/справа относительно центра.
    max_width = int(min(
        center_x - left_x0,
        right_x1 - center_x,
    ))

    if max_width < 10:
        return 0.0

    width = min(max_width, int(W * 0.30))

    lx0 = max(0, int(center_x - width))
    lx1 = int(center_x)

    rx0 = int(center_x)
    rx1 = min(W, int(center_x + width))

    left = bottom[y0:, lx0:lx1]
    right = bottom[y0:, rx0:rx1]

    if left.size == 0 or right.size == 0:
        return 0.0

    m = min(left.shape[1], right.shape[1])
    left = left[:, -m:]
    right = np.fliplr(right[:, :m])

    return float(np.clip((_safe_ncc(left, right) + 1.0) / 2.0, 0.0, 1.0))


def pair_score(
    left: dict,
    right: dict,
    side_score_left: float,
    side_score_right: float,
    center_x: float,
    roi_h: int,
) -> dict:
    """
    Оценка пары.

    Основные признаки:
      - обе стороны достаточно сильные;
      - одинаковый уровень по Y;
      - похожее расстояние от центра;
      - похожий размер.
    """

    dy = abs(left["cy"] - right["cy"])

    # 0 при большой разнице по Y, 1 при совпадении.
    y_sym = float(np.exp(-dy / max(1.0, roi_h * 0.12)))

    left_dx = abs(left["cx"] - center_x)
    right_dx = abs(right["cx"] - center_x)

    dx_diff = abs(left_dx - right_dx)
    dx_sym = float(np.exp(-dx_diff / max(1.0, center_x * 0.18)))

    area_ratio = min(
        left["area"],
        right["area"],
    ) / max(
        left["area"],
        right["area"],
        1,
    )

    # Сильная + слабая сторона разрешена.
    side_strength = 0.5 * (
        side_score_left + side_score_right
    )

    score = (
        0.40 * side_strength +
        0.25 * y_sym +
        0.20 * dx_sym +
        0.15 * area_ratio
    )

    return {
        "score": float(score),
        "dy": float(dy),
        "y_symmetry": y_sym,
        "x_symmetry": dx_sym,
        "area_ratio": float(area_ratio),
    }


def detect_pelvis_v2(
    image_path: str,

    # ROI
    bottom_frac: float = 0.48,

    # Центральная зона, которую не считаем тазом.
    center_gap_frac: float = 0.30,

    # Локальный контраст.
    local_sigma: float = 9.0,

    # Отдельный adaptive threshold для каждой стороны.
    mask_percentile: float = 55.0,
    min_abs_contrast: float = 2.0,

    # Фильтры компонентов.
    min_blob_area_frac: float = 0.0015,
    min_blob_width_frac: float = 0.025,
    min_blob_height_frac: float = 0.025,

    min_side_score: float = 0.43,
    min_pair_score: float = 0.52,
    weak_side_score: float = 0.38,
) -> dict:

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape

    # 1. Нормализация контраста
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    img_eq = clahe.apply(img)

    # Нижняя часть снимка.
    y0 = int(h * (1.0 - bottom_frac))
    bottom = img_eq[y0:].astype(np.float32)

    bh, bw = bottom.shape

    # 2. Локальный контраст
    local_contrast = make_local_contrast(
        bottom,
        sigma=local_sigma,
    )

    # 3. Центральная зона / приблизительная ось
    center_x = bw / 2.0

    center_half = int(bw * center_gap_frac / 2.0)

    left_x0 = 0
    left_x1 = max(1, int(center_x - center_half))

    right_x0 = min(bw - 1, int(center_x + center_half))
    right_x1 = bw

    # 4. Маски отдельно для L/R
    left_mask, left_thr = make_candidate_mask(
        local_contrast[:, left_x0:left_x1],
        threshold_percentile=mask_percentile,
        min_abs_contrast=min_abs_contrast,
    )

    right_mask, right_thr = make_candidate_mask(
        local_contrast[:, right_x0:right_x1],
        threshold_percentile=mask_percentile,
        min_abs_contrast=min_abs_contrast,
    )

    top_cut = int(bh * 0.40)

    left_mask[:top_cut] = 0
    right_mask[:top_cut] = 0

    # 5. Лучший компонент на каждой стороне
    left_blob = find_best_component(
        left_mask,
        local_contrast[:, left_x0:left_x1],
        x_offset=left_x0,
        roi_h=bh,
        roi_w=bw,
        min_area_frac=min_blob_area_frac,
        min_width_frac=min_blob_width_frac,
        min_height_frac=min_blob_height_frac,
    )

    right_blob = find_best_component(
        right_mask,
        local_contrast[:, right_x0:right_x1],
        x_offset=right_x0,
        roi_h=bh,
        roi_w=bw,
        min_area_frac=min_blob_area_frac,
        min_width_frac=min_blob_width_frac,
        min_height_frac=min_blob_height_frac,
    )

    # 6. Дополнительная evidence из нижних латеральных зон
    left_zone = lateral_zone_evidence(
        bottom,
        left_x0,
        left_x1,
    )

    right_zone = lateral_zone_evidence(
        bottom,
        right_x0,
        right_x1,
    )

    def side_total(blob, zone):
        comp_score = 0.0 if blob is None else blob["score"]

        # Component — основной признак.
        # Zone evidence — страховка для слабой кости.
        return float(
            0.65 * comp_score +
            0.35 * zone["score"]
        )

    left_score = side_total(left_blob, left_zone)
    right_score = side_total(right_blob, right_zone)

    # 7. Пара
    has_pair = False
    pair = None
    pair_info = None

    if (
        left_blob is not None
        and right_blob is not None
        and left_score >= min_side_score
        and right_score >= min_side_score
    ):
        pair_info = pair_score(
            left_blob,
            right_blob,
            left_score,
            right_score,
            center_x,
            bh,
        )

        if pair_info["score"] >= min_pair_score:
            has_pair = True
            pair = (left_blob, right_blob)

    # 8. Слабая сторона
    #
    # Если одна сторона слабее, но вторая уверенная,
    # разрешаем strong + weak только при наличии
    # дополнительной zone evidence на слабой стороне.
    strong_weak = False
    weak_pair_info = None

    if not has_pair:
        if left_blob is not None and right_blob is not None:
            if (
                left_score >= min_side_score
                and right_score >= weak_side_score
            ) or (
                right_score >= min_side_score
                and left_score >= weak_side_score
            ):
                weak_pair_info = pair_score(
                    left_blob,
                    right_blob,
                    left_score,
                    right_score,
                    center_x,
                    bh,
                )

                # Для strong+weak требуем немного более
                # выраженную геометрическую согласованность.
                if (
                    weak_pair_info["score"] >= min_pair_score
                    and weak_pair_info["y_symmetry"] >= 0.50
                ):
                    strong_weak = True
                    pair = (left_blob, right_blob)

    # 9. Мягкая симметрия всего нижнего ROI
    symmetry = mirror_symmetry_score(
        bottom,
        center_x=center_x,
        left_x0=left_x0,
        right_x1=right_x1,
        y_start_frac=0.55,
    )

    # 10. Финальное решение
    #
    # НЕ используем positive_frac как обязательный порог.
    # Он слишком сильно зависит от общей контрастности снимка.
    has_pelvis = bool(
        (has_pair or strong_weak)
    )

    # 11. Для отладки собираем результат
    result = {
        "has_pelvis": has_pelvis,

        "has_pair": bool(has_pair),
        "pair_type": (
            "strong_strong"
            if has_pair
            else ("strong_weak" if strong_weak else None)
        ),

        "pair_dy": (
            None
            if pair_info is None and weak_pair_info is None
            else float(
                (pair_info or weak_pair_info)["dy"]
            )
        ),

        "pair_score": (
            None
            if pair_info is None and weak_pair_info is None
            else float(
                (pair_info or weak_pair_info)["score"]
            )
        ),

        "symmetry": float(symmetry),

        "left_score": float(left_score),
        "right_score": float(right_score),

        "left_threshold": float(left_thr),
        "right_threshold": float(right_thr),

        "left_zone": left_zone,
        "right_zone": right_zone,

        "left_blob": left_blob,
        "right_blob": right_blob,

        "center_x": float(center_x),
        "roi_top_y": int(y0),
        "roi_height": int(bh),
        "roi_width": int(bw),
    }

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(os.path.abspath(image_path))
    vis_path = os.path.join(out_dir, f"{base}_vis.jpg")

    vis = cv2.cvtColor(bottom.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def draw_blob(blob, color, thickness=2):
        if blob is None:
            return
        x, y = int(blob["x"]), int(blob["y"])
        ww, hh = int(blob["w"]), int(blob["h"])
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), color, thickness)

    if pair is not None:
        draw_blob(pair[0], (0, 0, 255), 2)
        draw_blob(pair[1], (0, 0, 255), 2)

    cv2.imwrite(vis_path, vis)
    result["vis_path"] = vis_path

    return result


# Упрощённый запуск

def detect_pelvis(image_path: str) -> dict:
    return detect_pelvis_v2(image_path)

#
# if __name__ == "__main__":
#     import json
#
#     image_path = r"t829125.jpg"  # <-- поменяйте на свой путь
#
#     result = detect_pelvis(image_path)
#     print(json.dumps(result, indent=2, ensure_ascii=False))
