from PIL import Image
import numpy as np


class BlackPointsDetector:
    """
    Класс для поиска трёх чёрных точек под центром масс ярких точек
    на верхней половине изображения.

    Пример использования:
        detector = BlackPointsDetector('image.jpg', threshold=30)
        points = detector.get_points()
        new_points = detector.get_new_y_points()
    """

    def __init__(self, image_path, threshold=30, max_dist=20, n=3, x_tolerance=2):
        """
        :param image_path: путь к изображению
        :param threshold: порог яркости (чёрное <= threshold)
        :param max_dist: максимальное расстояние между соседними точками
        :param n: количество искомых чёрных точек
        :param x_tolerance: допуск по X вокруг центра масс
        """
        self.image_path = image_path
        self.threshold = threshold
        self.max_dist = max_dist
        self.n = n
        self.x_tolerance = x_tolerance

        # Загрузка изображения
        img = Image.open(image_path).convert('L')
        self.arr = np.array(img, dtype=np.uint8)
        self.height, self.width = self.arr.shape

        # Результаты (заполняются при вызове методов)
        self.points = None
        self.top_half = None
        self.mask = None
        self.cx = None
        self.cy = None
        self.y_cut = None
        self.blacks = None

    # ------------------------------------------------------------------
    # Внутренние шаги
    # ------------------------------------------------------------------
    def _get_bright_points(self):
        """Возвращает яркие точки на верхней половине изображения."""
        h = self.height
        top_half = self.arr[:h // 2, :]
        mask = top_half > self.threshold
        ys, xs = np.where(mask)
        values = top_half[ys, xs]
        points = np.column_stack((xs, ys, values))
        return points, top_half, mask

    def _center_of_mass_top30(self, points):
        """Центр масс по верхним 30% ярких точек."""
        xs = points[:, 0]
        ys = points[:, 1]
        vals = points[:, 2].astype(np.float64)

        y_min, y_max = ys.min(), ys.max()
        y_cut = y_min + 0.3 * (y_max - y_min + 1)
        sel = ys <= y_cut

        xs_sel, ys_sel, w_sel = xs[sel], ys[sel], vals[sel]
        total = w_sel.sum()
        cx = (xs_sel * w_sel).sum() / total
        cy = (ys_sel * w_sel).sum() / total
        return cx, cy, y_cut

    def _find_black_points_under(self, cx, cy):
        """
        Ищем n чёрных точек непосредственно под центром масс:
        - x в окне [cx - x_tolerance, cx + x_tolerance]
        - y > cy (ниже центра масс)
        - яркость <= threshold (чёрная зона)
        - попарное расстояние между выбранными точками <= max_dist
        """
        h, w = self.arr.shape
        arr = self.arr

        # Окно по X вокруг центра масс
        x_lo = max(0, int(np.floor(cx - self.x_tolerance)))
        x_hi = min(w, int(np.ceil(cx + self.x_tolerance)) + 1)

        # Нижняя область
        y0 = int(np.ceil(cy)) + 1
        if y0 >= h:
            return []

        # Чёрная маска в окне под центром масс
        black_mask = arr <= self.threshold
        sub = black_mask[y0:h, x_lo:x_hi]
        ys_r, xs_r = np.where(sub)

        if len(ys_r) == 0:
            return []

        # Абсолютные координаты
        ys_abs = ys_r + y0
        xs_abs = xs_r + x_lo

        candidates = np.column_stack((xs_abs, ys_abs)).astype(np.float64)

        # Сортируем по Y (сверху вниз)
        order = np.argsort(candidates[:, 1])
        candidates = candidates[order]

        # Непрерывный вертикальный «столб» чёрных пикселей
        chosen = []
        last_y = cy
        last_x = cx
        for x, y in candidates:
            if len(chosen) == 0:
                if y - last_y > self.max_dist:
                    continue
                chosen.append((x, y))
                last_y, last_x = y, x
            else:
                dy = y - last_y
                dx = abs(x - last_x)
                if dy <= 0:
                    continue
                if dy > self.max_dist or dx > self.max_dist:
                    continue
                chosen.append((x, y))
                last_y, last_x = y, x
                if len(chosen) == self.n:
                    break

        # Если не набрали n точек «цепочкой» — берём n самых верхних
        if len(chosen) < self.n:
            chosen = []
            ys_used = []
            for x, y in candidates:
                if all(abs(y - yy) > 2 for yy in ys_used):
                    chosen.append((x, y))
                    ys_used.append(y)
                    if len(chosen) == self.n:
                        break

        # Финальная проверка попарных расстояний
        result = []
        for x, y in chosen:
            ok = True
            for x2, y2 in result:
                if np.hypot(x - x2, y - y2) > self.max_dist:
                    ok = False
                    break
            if ok:
                result.append((x, y))
            if len(result) == self.n:
                break

        return [
            {'x': int(x), 'y': int(y),
             'intensity': int(arr[int(y), int(x)])}
            for x, y in result
        ]

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------
    def get_points(self):
        """
        Основной метод: возвращает список из n чёрных точек
        под центром масс ярких точек верхней половины изображения.

        :return: список словарей {'x': int, 'y': int, 'intensity': int}
        """
        # 1. Яркие точки
        self.points, self.top_half, self.mask = self._get_bright_points()

        # 2. Центр масс
        self.cx, self.cy, self.y_cut = self._center_of_mass_top30(self.points)

        # 3. Чёрные точки под центром масс
        self.blacks = self._find_black_points_under(self.cx, self.cy)
        return self.blacks

    def get_new_y_points(self, points=None):
        """
        Вычисляет новые Y-координаты для точек по формуле:
            new_y = (height - y) / 2 + y

        :param points: список точек (по умолчанию — результат get_points()).
                       Каждая точка — словарь с ключами 'x', 'y', 'intensity'.
        :return: новый список словарей с обновлёнными 'y' и добавленным 'old_y'.
        """
        if points is None:
            if self.blacks is None:
                self.get_points()
            points = self.blacks

        height = self.height
        new_points = []
        for p in points:
            old_y = p['y']
            new_y = (height - old_y) / 2 + old_y
            new_points.append({
                'x': p['x'],
                'y': new_y,
                'old_y': old_y,
                'intensity': p['intensity'],
            })
        return new_points

    # # ------------------------------------------------------------------
    # # Опциональная визуализация
    # # ------------------------------------------------------------------
    # def visualize(self, save_path='black_points_under_com.png', show=True):
    #     """Визуализация результата (требует matplotlib)."""
    #     try:
    #         import matplotlib.pyplot as plt
    #     except ImportError:
    #         print("matplotlib не установлен")
    #         return
    #
    #     if self.blacks is None:
    #         self.get_points()
    #
    #     fig, ax = plt.subplots(figsize=(8, 10))
    #     ax.imshow(self.arr, cmap='gray')
    #
    #     # Центр масс
    #     ax.plot(self.cx, self.cy, 'o', markersize=12, markerfacecolor='none',
    #             markeredgecolor='lime', markeredgewidth=2)
    #     ax.plot(self.cx, self.cy, '+', color='lime', markersize=15,
    #             markeredgewidth=2)
    #
    #     # Вертикаль через центр масс
    #     ax.axvline(self.cx, color='lime', lw=0.8, linestyle=':', alpha=0.7)
    #
    #     # Горизонтальная линия среза верхних 30%
    #     ax.axhline(self.y_cut, color='yellow', lw=1, linestyle='--')
    #
    #     # Чёрные точки
    #     for i, p in enumerate(new_points, 1):
    #         ax.plot(p['x'], p['y'], 'o', markersize=10,
    #                 markerfacecolor='red', markeredgecolor='white',
    #                 markeredgewidth=1.5)
    #         ax.text(p['x'] + 5, p['y'], f'{i}',
    #                 color='red', fontsize=12, fontweight='bold')
    #
    #     # Отрезки между точками
    #     for i in range(len(self.blacks) - 1):
    #         ax.plot([self.blacks[i]['x'], self.blacks[i + 1]['x']],
    #                 [self.blacks[i]['y'], self.blacks[i + 1]['y']],
    #                 color='red', lw=1.2, alpha=0.7)
    #
    #     ax.invert_yaxis()
    #     ax.set_title('Центр масс (зелёный) и чёрные точки под ним (красные)')
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=120)
    #     if show:
    #         plt.show()


# ----------------------------------------------------------------------
# Пример использования
# ----------------------------------------------------------------------
# if __name__ == '__main__':
#     detector = BlackPointsDetector(r'C:\Users\Хозяин\PycharmProjects\bones_hack\upload_dicom\2.25.13983650679559142983913935144968795415\series_003_2_CR\CR000001.jpg', threshold=30,
#                                    max_dist=70, n=3, x_tolerance=2)
#
#     # 1. Координаты трёх чёрных точек
#     points = detector.get_points()
#     print(f"Центр масс: cx={detector.cx:.2f}, cy={detector.cy:.2f}")
#     print(f"Найдено {len(points)} чёрных точек:")
#     for i, p in enumerate(points, 1):
#         print(f"  {i}) x={p['x']}, y={p['y']}, яркость={p['intensity']}")
#
#     # 2. Новые Y-координаты
#     new_points = detector.get_new_y_points()
#     print("\nНовые Y-координаты:")
#     for i, p in enumerate(new_points, 1):
#         print(f"  {i}) x={p['x']}, old_y={p['old_y']}, "
#               f"new_y={p['y']:.2f}, яркость={p['intensity']}")
#
#     # # 3. Визуализация
#     # detector.visualize()