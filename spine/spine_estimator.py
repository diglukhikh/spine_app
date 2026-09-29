import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # не визуализировать, только сохранять
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import os


class SpineAngleEstimator:
    """
    Оценка угла наклона позвоночника по изображению и JSON-маске (label='spine').

    Параметры конструктора:
        image_path : str  — путь к изображению
        json_path  : str  — путь к JSON с разметкой

    Метод estimate() возвращает:
        angle_deg : float — угол отклонения средней линии от вертикали (в градусах)
    """

    def __init__(self, image_path: str, json_path: str, save_path):
        self.image_path = image_path
        self.json_path = json_path

        # параметры (можно менять при желании, но по умолчанию работают)
        self.step_y = 1.0
        self.trim_ratio = 0.05
        self.smooth_window = 50
        self.save_path = save_path

        # поля-результаты
        self.img = None
        self.contour = None
        self.angle_deg = None
        self.angle_raw = None

    # ---------- 1. Загрузка ----------
    def _load_image(self):
        data = np.fromfile(self.image_path, dtype=np.uint8)  # np.fromfile ест юникод-пути
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.image_path)
        self.img = img

    def _load_contour(self):
        with open(self.json_path) as f:
            data = json.load(f)
        shape = next(s for s in data["shapes"] if s["label"] == "spine")
        self.contour = np.array(shape["points"], dtype=float)

    # ---------- 2. Построение параллельных линий ----------
    def _build_parallel_lines(self, r1=0.0, r2=1.0):
        contour = self.contour
        poly = Polygon(contour)
        if not poly.is_valid:
            poly = poly.buffer(0)

        y_min, y_max = contour[:, 1].min(), contour[:, 1].max()
        ys = np.arange(y_min, y_max + self.step_y, self.step_y)

        L1, L2 = [], []
        for y in ys:
            hline = LineString([(-1e6, y), (1e6, y)])
            inter = poly.intersection(hline)
            if inter.is_empty:
                continue
            if inter.geom_type == "LineString":
                coords = np.array(inter.coords)
            elif inter.geom_type == "MultiLineString":
                segs = [np.array(g.coords) for g in inter.geoms]
                coords = max(segs, key=lambda c: c[:, 0].max() - c[:, 0].min())
            else:
                continue

            xl, xr = coords[:, 0].min(), coords[:, 0].max()
            w = xr - xl
            L1.append([xl + w * r1, y])
            L2.append([xl + w * r2, y])
        return np.array(L1), np.array(L2)

    # ---------- 3. Обрезка краёв ----------
    @staticmethod
    def _trim_edges(line, trim_ratio=0.05):
        n = len(line)
        k = int(np.floor(n * trim_ratio))
        if 2 * k >= n:
            k = max(0, (n - 1) // 2)
        return line[k:n - k] if k > 0 else line

    # ---------- 4. Центральная линия ----------
    @staticmethod
    def _center_line(line1, line2):
        if len(line1) == len(line2):
            return np.column_stack([
                (line1[:, 0] + line2[:, 0]) / 2.0,
                line1[:, 1]
            ])
        y_common = np.linspace(
            max(line1[:, 1].min(), line2[:, 1].min()),
            min(line1[:, 1].max(), line2[:, 1].max()),
            500
        )
        x1 = np.interp(y_common, line1[:, 1], line1[:, 0])
        x2 = np.interp(y_common, line2[:, 1], line2[:, 0])
        return np.column_stack([(x1 + x2) / 2.0, y_common])

    # ---------- 5. Сглаживание ----------
    @staticmethod
    def _smooth_moving_average(line, window=31):
        if window < 3:
            return line.copy()
        if window % 2 == 0:
            window += 1
        window = min(window, len(line) if len(line) % 2 == 1 else len(line) - 1)
        if window < 3:
            return line.copy()

        kernel = np.ones(window) / window
        x_smooth = np.convolve(line[:, 0], kernel, mode="same")

        half = window // 2
        x_smooth[:half] = line[:half, 0]
        x_smooth[-half:] = line[-half:, 0]

        return np.column_stack([x_smooth, line[:, 1]])

    # ---------- 6. МНК ----------
    @staticmethod
    def _fit_angle_from_vertical(line):
        y = line[:, 1]
        x = line[:, 0]
        A = np.vstack([y, np.ones_like(y)]).T
        k, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return np.degrees(np.arctan(k)), k, b

    # ---------- 7. Визуализация с сохранением ----------
    def _save_plot(self, line1, line2, line1_trim, line2_trim,
                   center_line, center_line_smooth, k, b):
        fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                                 gridspec_kw={"width_ratios": [1, 1]})

        # 7.1 изображение
        ax = axes[0]
        ax.imshow(self.img, cmap="gray")
        ax.plot(self.contour[:, 0], self.contour[:, 1],
                "y--", lw=0.8, label="contour")
        ax.plot(line1[:, 0], line1[:, 1], "r-", lw=0.6, alpha=0.4)
        ax.plot(line2[:, 0], line2[:, 1], "b-", lw=0.6, alpha=0.4)
        ax.plot(line1_trim[:, 0], line1_trim[:, 1],
                "r-", lw=1.5, label="line 1 (90%)")
        ax.plot(line2_trim[:, 0], line2_trim[:, 1],
                "b-", lw=1.5, label="line 2 (90%)")
        ax.plot(center_line[:, 0], center_line[:, 1],
                "w-", lw=1.2, alpha=0.6, label="center (raw)")
        ax.plot(center_line_smooth[:, 0], center_line_smooth[:, 1],
                "lime", lw=2.2,
                label=f"center smoothed (w={self.smooth_window})")

        y_fit = np.array([center_line_smooth[:, 1].min(),
                          center_line_smooth[:, 1].max()])
        x_fit = k * y_fit + b
        ax.plot(x_fit, y_fit, "c--", lw=1.5,
                label=f"fit: {self.angle_deg:+.2f}\u00b0 от вертикали")

        ax.set_title("Осевые линии, средняя и сглаженная средняя")
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("off")

        # 7.2 X(Y)
        ax = axes[1]
        ax.plot(center_line[:, 0], center_line[:, 1],
                "w-", lw=1.0, alpha=0.6, label="center (raw)")
        ax.plot(center_line_smooth[:, 0], center_line_smooth[:, 1],
                "lime", lw=2.0,
                label=f"smoothed (w={self.smooth_window})")
        ax.plot(x_fit, y_fit, "c--", lw=1.5,
                label=f"fit: {self.angle_deg:+.2f}\u00b0")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("X, px")
        ax.set_ylabel("Y, px")
        ax.set_title("Средняя линия: raw vs smoothed")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        print("save_path =", repr(self.save_path))
        print("is dir:   ", os.path.isdir(self.save_path))
        print("exists dir:", os.path.exists(os.path.dirname(self.save_path)))
        plt.tight_layout()
        fig.savefig(self.save_path, dpi=150)
        plt.close(fig)

    # ---------- 8. Публичный метод ----------
    def estimate(self) -> float:
        """Возвращает угол отклонения средней линии от вертикали (в градусах)."""
        # 1. загрузка
        self._load_image()
        self._load_contour()

        # 2. параллельные линии
        line1, line2 = self._build_parallel_lines(r1=0.0, r2=1.0)

        # 3. обрезка
        line1_trim = self._trim_edges(line1, self.trim_ratio)
        line2_trim = self._trim_edges(line2, self.trim_ratio)

        # 4. средняя линия
        center_line = self._center_line(line1_trim, line2_trim)

        # 5. сглаживание
        center_line_smooth = self._smooth_moving_average(
            center_line, self.smooth_window
        )

        # 6. МНК
        self.angle_deg, k, b = self._fit_angle_from_vertical(center_line_smooth)
        self.angle_raw, _, _ = self._fit_angle_from_vertical(center_line)

        # 7. сохранение картинки
        self._save_plot(line1, line2, line1_trim, line2_trim,
                        center_line, center_line_smooth, k, b)

        return float(self.angle_deg)


# # ---------- пример использования ----------
# if __name__ == "__main__":
#     save_dir = r"C:\Users\Хозяин\PycharmProjects\bones_hack\results"
#     save_file = os.path.join(save_dir, "result.png")
#     estimator = SpineAngleEstimator(r"C:\Users\Хозяин\PycharmProjects\bones_hack\upload_dicom\upload_20260928_002141\CR000001.jpg", r"C:\Users\Хозяин\PycharmProjects\bones_hack\json\upload_20260928_002141_CR000001.json", save_file)
#     angle = estimator.estimate()
#     print(f"angle_deg = {angle:+.3f}")


