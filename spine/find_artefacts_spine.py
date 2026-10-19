import os
import numpy as np
import cv2


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread, работающий с не-ASCII путями (кириллица на Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    """cv2.imwrite, работающий с не-ASCII путями (кириллица на Windows)."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
    success, buf = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


class BrightAreaDetector:
    """
    Класс для поиска связных областей ярче заданного порога.
    """

    def __init__(self, threshold=250, min_area=300, max_area=None, name_start: str = "my",):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.name_start = name_start


    def _load_gray(self, image_path):
        img = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        return gray

    def _build_mask(self, gray):
        return (gray > self.threshold).astype(np.uint8)

    def _filter_components(self, mask):
        num, labeled, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        filtered = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, num):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            filtered[labeled == label] = 1

        return filtered

    def process(self, image_path, save_dir):
        gray = self._load_gray(image_path)
        if gray is None:
            return False

        mask = self._build_mask(gray)
        filtered = self._filter_components(mask)

        if not np.any(filtered):
            return False

        original = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if original is None:
            return False

        overlay = original.copy()
        overlay[filtered.astype(bool)] = (0, 0, 255)
        blended = cv2.addWeighted(original, 0.5, overlay, 0.5, 0)

        # создаём папку, если её нет (в т.ч. вложенную)
        os.makedirs(save_dir, exist_ok=True)

        # base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(save_dir, f"{self.name_start}_bright.png")

        return imwrite_unicode(out_path, blended)


if __name__ == "__main__":
    detector = BrightAreaDetector(threshold=253, min_area=300)
    result = detector.process("CR000003.jpg", r"C:\Users\Хозяин\PycharmProjects\bones_hack\spine\test")
    print(result)