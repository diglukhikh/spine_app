import cv2
import numpy as np
import json
import os
from scipy.signal import find_peaks


class SpineAnalyzer:
    """
    Анализ позвоночника на рентгеновском снимке.
    На вход: путь к JPG и путь к JSON с полигоном "spine".
    """

    # ---------- параметры по умолчанию ----------
    DEFAULTS = dict(
        smooth_win=17,          # окно сглаживания профиля по Y (нечётное)
        trend_win=36,           # окно вычитания медленного тренда (нечётное)
        min_period=20,          # мин. расстояние между пиками, px
        max_period=80,          # макс. расстояние между пиками, px
        peak_height_frac=0.38,  # мин. высота пика (доля от std)
        prominence_frac=0.38,   # мин. выраженность пика (доля от std)
    )

    def __init__(self, image_path: str, json_path: str, **kwargs):
        self.image_path = image_path
        self.json_path = json_path

        # применяем параметры: DEFAULTS, перекрытые kwargs
        self.params = {**self.DEFAULTS, **kwargs}

        # заполняется в analyze()
        self.image = None
        self.polygon = None
        self.spine_mask = None
        self.left = None
        self.right = None
        self.valid = None
        self.profile_detrended = None
        self.peaks = None
        self.vertebrae = []
        self.overlay = None

    # ================= публичный API =================

    def analyze(self) -> dict:
        """Запускает весь конвейер и возвращает результат."""
        self._load()
        self._prepare_mask()
        self._build_profile()
        self._find_peaks()
        self._build_vertebrae()
        self._make_overlay()
        return self.result()

    def result(self) -> dict:
        return {
            "n_vertebrae": len(self.vertebrae),
            "vertebrae": self.vertebrae,
            "overlay": self.overlay,
        }

    def save_overlay(self, out_dir: str = "vertebrae_debug") -> str:
        """Сохраняет overlay в файл, возвращает путь."""
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_overlay.jpg")
        cv2.imwrite(out_path, self.overlay)
        return out_path

    # ================= шаги конвейера =================

    def _load(self):
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(self.image_path)

        self.polygon = self._load_polygon(self.json_path)

    @staticmethod
    def _load_polygon(json_path: str) -> np.ndarray:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if shape.get("label") == "spine":
                pts = np.array(shape["points"], dtype=np.float32)
                return pts.astype(np.int32)
        raise ValueError("no 'spine' polygon in JSON")

    def _prepare_mask(self):
        h, w = self.image.shape
        self.spine_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.spine_mask, [self.polygon], 1)

        # CLAHE — локальное выравнивание контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.image_eq = clahe.apply(self.image)

    def _build_profile(self):
        h, w = self.image.shape

        # границы полигона по строкам
        left = np.full(h, np.nan, dtype=np.float32)
        right = np.full(h, np.nan, dtype=np.float32)
        valid = np.zeros(h, dtype=bool)

        for y in range(h):
            idx = np.where(self.spine_mask[y] > 0)[0]
            if idx.size < 4:
                continue
            left[y] = idx.min()
            right[y] = idx.max() + 1
            valid[y] = True

        self.left = left
        self.right = right
        self.valid = valid

        # профиль яркости вдоль оси
        profile = np.full(h, np.nan, dtype=np.float32)
        for y in range(h):
            if not valid[y]:
                continue
            l = int(max(0, left[y]))
            r = int(min(w, right[y]))
            if r - l < 4:
                continue
            profile[y] = self.image_eq[y, l:r].mean()

        # интерполяция NaN
        if valid.sum() > 0:
            idx = np.arange(h)
            profile[~valid] = np.interp(idx[~valid], idx[valid], profile[valid])

        # сглаживание
        sw = self.params["smooth_win"]
        if sw % 2 == 0:
            sw += 1
        if sw >= 3:
            kernel = np.ones(sw, dtype=np.float32) / sw
            profile = np.convolve(profile, kernel, mode="same")

        # вычитание тренда
        tw = self.params["trend_win"]
        if tw % 2 == 0:
            tw += 1
        kernel = np.ones(tw, dtype=np.float32) / tw
        trend = np.convolve(profile, kernel, mode="same")
        self.profile_detrended = profile - trend

    def _find_peaks(self):
        ys = np.where(self.valid)[0]
        if ys.size < 20:
            self.peaks = np.array([], dtype=int)
            return

        y0, y1 = int(ys.min()), int(ys.max())
        seg = self.profile_detrended[y0:y1 + 1]

        pos = seg[seg > 0]
        if pos.size == 0:
            self.peaks = np.array([], dtype=int)
            return

        p = self.params
        height_thr = max(pos.std() * p["peak_height_frac"],
                         pos.max() * p["peak_height_frac"] * 0.5)
        prominence_thr = pos.std() * p["prominence_frac"]

        peaks, _ = find_peaks(
            seg,
            distance=p["min_period"],
            height=height_thr,
            prominence=prominence_thr,
        )
        if peaks.size == 0:
            self.peaks = np.array([], dtype=int)
            return

        # фильтр по периоду + достройка пропущенных
        peaks_global = peaks + y0
        filtered = [peaks_global[0]]
        for pk in peaks_global[1:]:
            if pk - filtered[-1] <= p["max_period"]:
                filtered.append(pk)
            else:
                n_missing = max(1, round((pk - filtered[-1]) / p["max_period"]))
                step = (pk - filtered[-1]) / (n_missing + 1)
                for k in range(1, n_missing + 1):
                    filtered.append(int(filtered[-1] + step))
                filtered.append(pk)

        self.peaks = np.array(sorted(set(filtered)), dtype=int)

    def _build_vertebrae(self):
        peaks = self.peaks
        if peaks.size == 0:
            self.vertebrae = []
            return

        h = self.image.shape[0]
        blobs = []
        for i, cy in enumerate(peaks):
            y_top = int((peaks[i - 1] + cy) / 2) if i > 0 else 0
            y_bot = int((peaks[i + 1] + cy) / 2) if i < len(peaks) - 1 else h - 1

            l = int(self.left[cy])
            r = int(self.right[cy])

            blobs.append({
                "id": i + 1,
                "cy": int(cy),
                "cx": int((l + r) / 2),
                "x": l,
                "y": y_top,
                "w": r - l,
                "h": y_bot - y_top,
            })
        self.vertebrae = blobs

    def _make_overlay(self):
        """Всегда строит только overlay (исходник + номера позвонков)."""
        overlay = cv2.cvtColor(self.image_eq, cv2.COLOR_GRAY2BGR)
        cv2.polylines(overlay, [self.polygon], True, (0, 200, 0), 2)

        for b in self.vertebrae:
            cv2.rectangle(overlay,
                          (b["x"], b["y"]),
                          (b["x"] + b["w"], b["y"] + b["h"]),
                          (0, 255, 255), 1)
            cv2.circle(overlay, (b["cx"], b["cy"]), 3, (0, 0, 255), -1)
            cv2.putText(overlay, str(b["id"]),
                        (b["x"], max(12, b["y"] + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
        self.overlay = overlay


# ================= запуск =================

# if __name__ == "__main__":
#     image_path = r"upload_dicom/upload_20260927_222413/CR000001.jpg"
#     json_path = r"json/upload_20260927_222413_CR000001.json"
#
#     analyzer = SpineAnalyzer(image_path, json_path)
#     result = analyzer.analyze()
#
#     print(f"{image_path}: n_vertebrae = {result['n_vertebrae']}")
#     for b in result["vertebrae"]:
#         print(f"  #{b['id']}: center=({b['cx']},{b['cy']}) "
#               f"bbox=({b['x']},{b['y']},{b['w']}x{b['h']})")
#
#     path = analyzer.save_overlay()
#     print(f"overlay saved: {path}")