import cv2
import numpy as np

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread, работающий с не-ASCII путями (кириллица на Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

class FragmentExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        # self.image = imread_unicode(image_path)
        self.img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Не удалось открыть {image_path}")

    # ------------------------------------------------------------------ #
    # 1. Вырезание области интереса (side=1 — left, side=2 — right)
    # ------------------------------------------------------------------ #
    def crop_int(self, side):
        h, w = self.img.shape[:2]
        if side == 1:
            y1, y2 = int(h * 0.05), int(h * 0.4)
            x1, x2 = int(w * 0.05), int(w * 0.35)
            name = "left"
        elif side == 2:
            y1, y2 = int(h * 0.05), int(h * 0.4)
            x1, x2 = int(w * 0.65), int(w * 0.95)
            name = "right"
        else:
            raise ValueError("side must be 1 or 2")

        crop = self.img[y1:y2, x1:x2]
        crop_coords = (x1, y1, x2, y2)  # (x1, y1, x2, y2) в исходном изображении

        mean_intensity = crop.mean()
        max_intensity = crop.max()

        threshold = max_intensity * 0.7
        bright_pixels = np.count_nonzero(crop > threshold)
        total_pixels = crop.size
        percent = bright_pixels / total_pixels * 100

        if percent < 2:
            low_bound = np.maximum(max_intensity * 0.6, 80)
            area_bound = 10
        elif 2 < percent < 5:
            low_bound = np.maximum(max_intensity * 0.8, 80)
            area_bound = 20
        else:
            low_bound = max_intensity * 0.95
            area_bound = 30

        return {
            "low_bound": low_bound,
            "area_bound": area_bound,
            "name": name,
            "crop": crop,
            "crop_coords": crop_coords,  # (x1, y1, x2, y2)
        }

    # ------------------------------------------------------------------ #
    # 2. Поиск ярких связных компонент
    # ------------------------------------------------------------------ #
    def find_bright_components(self, crop, low_bound, area_bound, top_n=4):
        blur = cv2.GaussianBlur(crop, (3, 3), 0)
        _, thresh = cv2.threshold(blur, low_bound, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            mask = (labels == i)
            mean_brightness = blur[mask].mean()
            max_brightness = blur[mask].max()
            if 800 > area > area_bound:
                components.append({
                    'id': i, 'area': area, 'bbox': (x, y, w, h),
                    'mean': mean_brightness, 'max': max_brightness,
                    'centroid': centroids[i]
                })

        components.sort(key=lambda c: c['mean'], reverse=True)
        return components[:top_n]

    # ------------------------------------------------------------------ #
    # 3. Объединение и расширение компонент
    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_and_expand_components(components, img_shape, target_size=48, max_components=4):
        img_h, img_w = img_shape[:2]

        if len(components) > max_components:
            components = sorted(components, key=lambda c: c['area'], reverse=True)[:max_components]

        def bbox_center(bbox):
            x, y, w, h = bbox
            return (x + w / 2, y + h / 2)

        def distance(c1, c2):
            cx1, cy1 = bbox_center(c1['bbox'])
            cx2, cy2 = bbox_center(c2['bbox'])
            return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        result = [dict(c) for c in components]

        while len(result) > 1:
            min_dist = float('inf')
            merge_pair = None
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    d = distance(result[i], result[j])
                    if d < min_dist:
                        min_dist = d
                        merge_pair = (i, j)

            if len(result) <= max_components and min_dist > target_size:
                break

            i, j = merge_pair
            c1, c2 = result[i], result[j]

            x1, y1, w1, h1 = c1['bbox']
            x2, y2, w2, h2 = c2['bbox']

            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_x2 = max(x1 + w1, x2 + w2)
            new_y2 = max(y1 + h1, y2 + h2)
            new_w = new_x2 - new_x
            new_h = new_y2 - new_y

            merged = {
                'id': c1['id'],
                'area': c1['area'] + c2['area'],
                'bbox': (new_x, new_y, new_w, new_h),
                'mean': (c1['mean'] * c1['area'] + c2['mean'] * c2['area']) / (c1['area'] + c2['area']),
                'max': max(c1['max'], c2['max']),
                'centroid': (
                    (c1['centroid'][0] * c1['area'] + c2['centroid'][0] * c2['area']) / (c1['area'] + c2['area']),
                    (c1['centroid'][1] * c1['area'] + c2['centroid'][1] * c2['area']) / (c1['area'] + c2['area'])
                )
            }

            result = [r for k, r in enumerate(result) if k != i and k != j]
            result.append(merged)

        final_bboxes = []
        for comp in result:
            x, y, w, h = comp['bbox']
            cx, cy = x + w / 2, y + h / 2

            new_x = cx - target_size / 2
            new_y = cy - target_size / 2

            new_x = max(0, min(new_x, img_w - target_size))
            new_y = max(0, min(new_y, img_h - target_size))

            final_w = min(target_size, img_w)
            final_h = min(target_size, img_h)

            if new_x + final_w > img_w:
                new_x = img_w - final_w
            if new_y + final_h > img_h:
                new_y = img_h - final_h

            final_bboxes.append({
                'id': comp['id'],
                'area': comp['area'],
                'bbox': (int(new_x), int(new_y), int(final_w), int(final_h)),
                'mean': comp['mean'],
                'max': comp['max'],
                'centroid': comp['centroid']
            })

        return final_bboxes

    # ------------------------------------------------------------------ #
    # 4. Полный пайплайн для одной стороны
    # ------------------------------------------------------------------ #
    def process_side(self, side, top_n=4, target_size=48, max_components=4):
        # Шаг 1: вырезание области
        crop_info = self.crop_int(side)
        crop = crop_info['crop']
        crop_coords = crop_info['crop_coords']  # (x1, y1, x2, y2)

        # Шаг 2: поиск ярких компонент
        brightest = self.find_bright_components(
            crop, crop_info['low_bound'], crop_info['area_bound'], top_n=top_n
        )

        # Шаг 3: объединение и расширение
        img_shape = crop.shape[:2]
        merged = self.merge_and_expand_components(
            brightest, img_shape, target_size=target_size, max_components=max_components
        )

        # Шаг 4: перевод локальных координат в координаты исходного изображения
        # merged bbox = (x, y, w, h) в системе crop
        # crop_coords = (x1, y1, x2, y2) в системе исходного изображения
        merged_global = []
        for m in merged:
            x, y, w, h = m['bbox']
            gx = x + crop_coords[0]
            gy = y + crop_coords[1]
            merged_global.append({
                'bbox_global': (gx, gy, w, h),  # (x, y, w, h) в исходном изображении
                'bbox_global_x1y1x2y2': (gx, gy, gx + w, gy + h),  # (x1, y1, x2, y2)
                'area': m['area'],
                'mean': m['mean'],
                'max': m['max'],
                'id': m['id'],
            })

        return {
            'side': side,
            'name': crop_info['name'],
            'crop_coords': crop_coords,        # (x1, y1, x2, y2) в исходном изображении
            'merged_bboxes': merged_global,
        }

    # ------------------------------------------------------------------ #
    # 5. Запуск для обеих сторон
    # ------------------------------------------------------------------ #
    def run(self):
        results = {}
        for side in (1, 2):
            results[side] = self.process_side(side)
        return results

# image_path = r"C:\Users\Хозяин\PycharmProjects\bones_hack\operation\CR000001_without_object_filled.png"
# extractor = FragmentExtractor(image_path)
# results = extractor.run()

# import os, cv2
# p = r"C:\Users\Хозяин\PycharmProjects\bones_hack\operation\CR000001_without_object_filled.png"
#
# print("exists:", os.path.exists(p))
# print("cv2.imread:", cv2.imread(p))  # скорее всего None
# print("imdecode:", cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR))