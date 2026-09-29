import cv2
import numpy as np


class SpineAligner:
    """
    Класс для проверки и центрирования позвоночника на рентгеновском снимке.

    Использование:
        aligner = SpineAligner()
        info = aligner.process("/path/to/image.jpg")
    """

    def __init__(self,
                 threshold_percentile=90,
                 center_zone=0.10,
                 min_pixels=50,
                 method="mass",
                 border_mode=cv2.BORDER_REPLICATE,
                 resize_to=(300, 300)):
        """
        Parameters
        ----------
        threshold_percentile : float
            Процентиль яркости для выделения позвоночника.
        center_zone : float
            Ширина центральной зоны (доля от ширины изображения).
        min_pixels : int
            Минимальное число пикселей маски (защита от мусора).
        method : str
            "mass" — центр масс, "fitline" — аппроксимация прямой.
        border_mode : int
            Режим границы для warpAffine.
        resize_to : tuple | None
            Размер для финального ресайза. None — не ресайзить.
        """
        self.threshold_percentile = threshold_percentile
        self.center_zone = center_zone
        self.min_pixels = min_pixels
        self.method = method
        self.border_mode = border_mode
        self.resize_to = resize_to

    # ------------------------------------------------------------------ #
    def _extract_spine_mask(self, gray):
        """Возвращает маску позвоночника (крупнейшая яркая компонента)."""
        thr = np.percentile(gray, self.threshold_percentile)
        mask = (gray >= thr).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        clean = (labels == best).astype(np.uint8)
        ys, xs = np.nonzero(clean)
        if xs.size < self.min_pixels:
            return None, None
        return clean, (xs, ys)

    # ------------------------------------------------------------------ #
    def _find_axis_x(self, clean, xs, ys, w, h):
        """Возвращает X-координату оси позвоночника."""
        if self.method == "mass":
            return float(xs.mean())
        # fitline
        cnts, _ = cv2.findContours(clean * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        mid_y = h / 2.0
        t = (mid_y - y0) / vy if abs(vy) > 1e-6 else 0.0
        return float(x0 + vx * t)

    # ------------------------------------------------------------------ #
    def process(self, image_path):
        """
        Обрабатывает изображение: при необходимости центрирует позвоночник
        и делает ресайз. Файл перезаписывается по тому же пути.

        Returns
        -------
        info : dict
            {
                "ok": bool,
                "centered": bool,       # был ли позвоночник уже по центру
                "shifted": bool,        # было ли выполнено смещение
                "cx_before_norm": float,
                "cx_after_norm": float,
                "shift_px": float,
                "width": int, "height": int,
                "reason": str | None,
            }
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "reason": "cannot read image",
                    "centered": False, "shifted": False}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clean, pts = self._extract_spine_mask(gray)
        if clean is None:
            return {"ok": False, "reason": "no spine found",
                    "centered": False, "shifted": False,
                    "width": w, "height": h}

        xs, ys = pts
        cx = self._find_axis_x(clean, xs, ys, w, h)

        left = (0.5 - self.center_zone / 2) * w
        right = (0.5 + self.center_zone / 2) * w
        centered = left <= cx <= right

        info = {
            "ok": True,
            "centered": bool(centered),
            "shifted": False,
            "cx_before_norm": cx / w,
            "cx_after_norm": cx / w,
            "shift_px": 0.0,
            "width": w,
            "height": h,
            "reason": None,
        }

        # --- смещение только если НЕ по центру ---
        if not centered:
            target_cx = w / 2.0
            dx = target_cx - cx
            M = np.float32([[1, 0, dx],
                            [0, 1, 0]])
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=self.border_mode,
                borderValue=(0, 0, 0)
            )
            info["shifted"] = True
            info["shift_px"] = float(dx)
            info["cx_after_norm"] = (cx + dx) / w

        # --- ресайз в любом случае ---
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_AREA)

        # --- перезапись исходного файла ---
        cv2.imwrite(image_path, img)

        return info


# ====================================================================== #
if __name__ == "__main__":
    aligner = SpineAligner(
        threshold_percentile=70,
        center_zone=0.10,
        method="mass",
    )
    info = aligner.process("upload_dicom/upload_20260928_002141/CR000001.jpg")

    if not info["ok"]:
        print("Ошибка:", info["reason"])
    elif info["centered"]:
        print("ПОЗВОНОЧНИК ПО ЦЕНТРУ (смещение не требуется)")
        print(f"cx = {info['cx_before_norm'] * 100:.1f}% ширины")
    else:
        print("ПОЗВОНОЧНИК СМЕЩЁН — изображение отцентрировано")
        print(f"cx было:  {info['cx_before_norm'] * 100:.1f}%")
        print(f"сдвиг:    {info['shift_px']:+.0f} px")
        print(f"cx стало: {info['cx_after_norm'] * 100:.1f}%")