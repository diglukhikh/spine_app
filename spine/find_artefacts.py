# @title
import os
import cv2
import numpy as np


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


class BrightComponentsDetector:
    """
    Ищет яркие связные компоненты внутри ROI на двух верхних фрагментах изображения:
      - левый верхний угол: 45% высоты, 47% ширины
      - правый верхний угол: 45% высоты, 47% ширины

    Возвращает True, если хотя бы на одном фрагменте найдены компоненты.

    Пример:
        det = BrightComponentsDetector()
        print(det.detect("path/to/image.jpg"))

        det = BrightComponentsDetector(save_artifacts=True,
                                       artifacts_dir=".../artefacts_img")
        det.detect("path/to/image.jpg")
    """

    def __init__(
        self,
        save_artifacts: bool = False,
        artifacts_dir: str = "test",
        name_start: str = "my",
        # --- фрагменты ---
        fragment_h_ratio: float = 0.45,
        fragment_w_ratio: float = 0.47,
        # --- параметры Canny ---
        canny_low: int = 80,
        canny_high: int = 150,
        # --- параметры ROI ---
        allowed_margin_ratio: float = 0.02,
        min_area: float = 50,
        max_area: float = 10000,
        roi_dilate: int = 5,
        # --- параметры порога яркости ---
        low_bound_min: float = 80,
        low_bound_max: float = 120,
        low_bound_ratio: float = 0.7,
        # --- параметры фильтрации компонент ---
        contrast_bound_large: float = 85,
        contrast_bound_small: float = 65,
        total_len_threshold: float = 600.0,
        # --- параметры подсветки артефакта ---
        highlight_expand_px: int = 30,
    ):
        self.save_artifacts = save_artifacts
        self.artifacts_dir = artifacts_dir
        self.name_start = name_start

        self.fragment_h_ratio = fragment_h_ratio
        self.fragment_w_ratio = fragment_w_ratio

        self.canny_low = canny_low
        self.canny_high = canny_high

        self.allowed_margin_ratio = allowed_margin_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.roi_dilate = roi_dilate

        self.low_bound_min = low_bound_min
        self.low_bound_max = low_bound_max
        self.low_bound_ratio = low_bound_ratio

        self.contrast_bound_large = contrast_bound_large
        self.contrast_bound_small = contrast_bound_small
        self.total_len_threshold = total_len_threshold

        self.highlight_expand_px = highlight_expand_px

    # ------------------------------------------------------------------
    # Вспомогательные метрики
    # ------------------------------------------------------------------
    @staticmethod
    def _sobel_mag(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _sharpness_gradient_norm(comp_mask, gray_img, grad):
        k = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(comp_mask, k) - cv2.erode(comp_mask, k)
        if boundary.sum() == 0:
            return 0.0, 0.0

        inner = cv2.erode(comp_mask, np.ones((3, 3), np.uint8))
        ring_out = cv2.dilate(comp_mask, np.ones((7, 7), np.uint8)) - cv2.dilate(comp_mask, k)

        if inner.sum() == 0 or ring_out.sum() == 0:
            return 0.0, 0.0

        contrast = abs(gray_img[inner > 0].mean() - gray_img[ring_out > 0].mean())
        g_mean = grad[boundary > 0].mean()
        sharp_A = g_mean / (contrast + 1e-6) if contrast > 1e-3 else 0.0
        return float(sharp_A), float(contrast)

    # ------------------------------------------------------------------
    # Внутренние шаги алгоритма (на одном фрагменте)
    # ------------------------------------------------------------------
    def _build_allowed_mask(self, gray):
        _, black_mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV)
        h, w = gray.shape[:2]
        margin = int(round(self.allowed_margin_ratio * max(h, w)))

        mask_contours, _ = cv2.findContours(
            black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        forbidden = np.zeros_like(gray)
        cv2.drawContours(forbidden, mask_contours, -1, 255, thickness=cv2.FILLED)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
        forbidden_dilated = cv2.dilate(forbidden, kernel)
        allowed = cv2.bitwise_not(forbidden_dilated)
        return allowed

    def _filter_contours(self, gray, allowed):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours_mask = np.zeros_like(gray)
        cv2.drawContours(contours_mask, contours, -1, 255, thickness=2)
        clipped_mask = cv2.bitwise_and(contours_mask, allowed)

        clipped_contours, _ = cv2.findContours(
            clipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = [
            c for c in clipped_contours
            if self.min_area <= cv2.contourArea(c) <= self.max_area
        ]

        total_len = sum(cv2.arcLength(c, True) for c in filtered_contours)
        return filtered_contours, total_len

    def _build_roi_mask(self, gray_shape, filtered_contours):
        roi_mask = np.zeros(gray_shape, dtype=np.uint8)
        cv2.drawContours(roi_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
        kernel_roi = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.roi_dilate + 1, 2 * self.roi_dilate + 1),
        )
        roi_mask = cv2.dilate(roi_mask, kernel_roi)
        return roi_mask

    def _find_components(self, gray, roi_mask, total_len):
        blur = gray

        max_intensity = gray.max()
        low_bound = max(max_intensity * self.low_bound_ratio, self.low_bound_min)
        low_bound = min(low_bound, self.low_bound_max)

        _, thresh = cv2.threshold(blur, low_bound, 255, cv2.THRESH_BINARY)
        thresh_roi = cv2.bitwise_and(thresh, roi_mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh_roi, connectivity=8
        )

        grad = self._sobel_mag(blur)

        if total_len > self.total_len_threshold:
            contrast_bound = self.contrast_bound_large
        else:
            contrast_bound = self.contrast_bound_small

        components = []
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            mask = (labels == i)
            if not mask.any():
                continue

            mean_brightness = blur[mask].mean()
            max_brightness = blur[mask].max()

            comp_mask = mask.astype(np.uint8)
            _, contrast = self._sharpness_gradient_norm(comp_mask, blur, grad)

            keep = False
            if (
                mean_brightness < 130
                and 25 < area < 350
                and contrast > contrast_bound - 20
            ):
                keep = True
            elif 20 < area < 350 and contrast > contrast_bound:
                keep = True
            elif max_brightness > 210 and contrast > contrast_bound:
                keep = True

            if keep:
                components.append({
                    "id": i,
                    "area": area,
                    "bbox": (x, y, ww, hh),
                    "mean": float(mean_brightness),
                    "max": float(max_brightness),
                    "centroid": centroids[i],
                    "contrast": float(contrast),
                })

        components.sort(key=lambda c: c["mean"], reverse=True)
        return components, labels

    # ------------------------------------------------------------------
    # Сохранение артефакта для одного фрагмента
    # ------------------------------------------------------------------
    def _save_artifact(self, img_bgr, roi_mask, labels, components, out_path):
        """Подсвечивает локальный ROI (в пределах highlight_expand_px от компонент)."""
        h, w = img_bgr.shape[:2]

        comp_mask = np.zeros((h, w), dtype=np.uint8)
        for c in components:
            comp_mask[labels == c["id"]] = 255

        if self.highlight_expand_px > 0:
            k = 2 * self.highlight_expand_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            expanded = cv2.dilate(comp_mask, kernel)
        else:
            expanded = comp_mask.copy()

        local_roi = cv2.bitwise_and(roi_mask, expanded)

        result = img_bgr.copy()

        overlay = result.copy()
        overlay[local_roi > 0] = (0, 255, 255)          # жёлтый
        result = cv2.addWeighted(overlay, 0.30, result, 0.70, 0)

        overlay2 = result.copy()
        overlay2[comp_mask > 0] = (0, 0, 255)           # красный
        result = cv2.addWeighted(overlay2, 0.55, result, 0.45, 0)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, result)

    # ------------------------------------------------------------------
    # Обработка одного фрагмента
    # ------------------------------------------------------------------
    def _process_fragment(self, fragment_bgr, artifact_path=None):
        """Возвращает True, если на фрагменте найдены компоненты."""
        if fragment_bgr is None or fragment_bgr.size == 0:
            return False

        gray = cv2.cvtColor(fragment_bgr, cv2.COLOR_BGR2GRAY)

        allowed = self._build_allowed_mask(gray)
        filtered_contours, total_len = self._filter_contours(gray, allowed)
        roi_mask = self._build_roi_mask(gray.shape, filtered_contours)
        components, labels = self._find_components(gray, roi_mask, total_len)

        self._save_artifact(fragment_bgr, roi_mask, labels, components, artifact_path)

        # if self.save_artifacts and artifact_path is not None and components:
        #     self._save_artifact(fragment_bgr, roi_mask, labels, components, artifact_path)

        return len(components) > 0

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def detect(self, image_path: str) -> bool:
        """
        Читает изображение, вырезает два верхних фрагмента (левый и правый),
        прогоняет каждый через алгоритм поиска ярких компонент.
        Возвращает True, если хотя бы на одном фрагменте что-то найдено.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        img = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        h, w = img.shape[:2]
        fh = int(round(h * self.fragment_h_ratio))
        fw = int(round(w * self.fragment_w_ratio))

        # Левый верхний угол
        left_frag = img[0:fh, 0:fw].copy()
        # Правый верхний угол
        right_frag = img[0:fh, w - fw:w].copy()

        base = os.path.splitext(os.path.basename(image_path))[0]

        left_path = (
            os.path.join(self.artifacts_dir, f"{self.name_start}_left.jpg")
            if self.save_artifacts else None
        )
        right_path = (
            os.path.join(self.artifacts_dir, f"{self.name_start}_right.jpg")
            if self.save_artifacts else None
        )

        left_has = self._process_fragment(left_frag, left_path)
        right_has = self._process_fragment(right_frag, right_path)

        return bool(left_has or right_has)


# ----------------------------------------------------------------------
# Пример использования
# ----------------------------------------------------------------------
if __name__ == "__main__":
    detector = BrightComponentsDetector(
        save_artifacts=True,
        artifacts_dir="test",
    )
    result = detector.detect(r"C:\Users\Хозяин\PycharmProjects\bones_hack\spine\CR000004_without_object_filled.jpg")
    print(result)