import json
import numpy as np
import chardet
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from matplotlib.path import Path


class ContourAreaDiff:
    """
    Класс для вычисления:
      1) разницы площадей между аппроксимированной сплайном 4-й степени
         кривой и исходными точками контура (правая часть контура
         относительно центра масс) до общей хорды;
      2) метрик расстояния между этими двумя кривыми (mean, median, rms,
         hausdorff и т.д.).
    """

    def __init__(self, json_path):
        self.json_path = json_path
        self.pts = None
        self.H = None
        self.W = None
        self.mean = None
        self.x_r = None
        self.y_r = None
        self.point_min_y = None

    # ------------------------------------------------------------
    # 1. Загрузка JSON
    # ------------------------------------------------------------
    def load(self):
        with open(self.json_path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        text = raw.decode(detected["encoding"])
        data = json.loads(text)

        shape = data["shapes"][0]
        self.pts = np.array(shape["points"], dtype=float)
        self.H = data["imageHeight"]
        self.W = data["imageWidth"]
        return self

    # ------------------------------------------------------------
    # 2. Центр масс контура
    # ------------------------------------------------------------
    def compute_centroid(self):
        poly_path = Path(self.pts)
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = poly_path.contains_points(grid_pts).reshape(self.H, self.W)
        ys, xs = np.where(inside)
        pts_inside = np.column_stack([xs, ys]).astype(float)
        self.mean = pts_inside.mean(axis=0)
        return self

    # ------------------------------------------------------------
    # 3. Выделение правой части контура + фильтрация
    # ------------------------------------------------------------
    def extract_right_contour(self):
        right_mask = self.pts[:, 0] >= self.mean[0]
        right_pts = self.pts[right_mask]

        order = np.argsort(right_pts[:, 1])
        right_pts = right_pts[order]

        if len(right_pts) > 0:
            idx_min_y = np.argmin(right_pts[:, 1])
            self.point_min_y = right_pts[idx_min_y]
            keep_mask = right_pts[:, 0] >= self.point_min_y[0]
            right_pts = right_pts[keep_mask]
        else:
            self.point_min_y = None

        self.x_r = right_pts[:, 0]
        self.y_r = right_pts[:, 1]
        return self

    # ------------------------------------------------------------
    # 4. Построение сплайна степени k
    # ------------------------------------------------------------
    @staticmethod
    def build_spline(x_r, y_r, s_factor, k=4, n_fine=5000):
        t = np.zeros(len(x_r))
        t[1:] = np.cumsum(np.hypot(np.diff(x_r), np.diff(y_r)))
        t /= t[-1]
        k_eff = min(k, len(x_r) - 1)
        spl, u = splprep([x_r, y_r], u=t, s=len(x_r) * s_factor, k=k_eff)
        u_fine = np.linspace(0, 1, n_fine)
        x_fine, y_fine = splev(u_fine, spl)
        return spl, u_fine, x_fine, y_fine

    # ------------------------------------------------------------
    # 5. Площадь между кривой и хордой
    # ------------------------------------------------------------
    @staticmethod
    def area_curve_to_chord(x_fine, y_fine):
        chord_x = np.array([x_fine[-1], x_fine[0]])
        chord_y = np.array([y_fine[-1], y_fine[0]])
        px = np.concatenate([x_fine, chord_x])
        py = np.concatenate([y_fine, chord_y])
        area_signed = 0.5 * np.sum(px[:-1] * py[1:] - px[1:] * py[:-1])
        return abs(area_signed)

    # ------------------------------------------------------------
    # 6. Метрики расстояний между кривыми
    # ------------------------------------------------------------
    @staticmethod
    def _curve_distances(x1, y1, x2, y2):
        p1 = np.column_stack([x1, y1])
        p2 = np.column_stack([x2, y2])
        tree = cKDTree(p2)
        d, _ = tree.query(p1, k=1)
        return d

    @classmethod
    def compare_curves(cls, xA, yA, xB, yB, nameA="A", nameB="B"):
        dAB = cls._curve_distances(xA, yA, xB, yB)   # A → B
        dBA = cls._curve_distances(xB, yB, xA, yA)   # B → A
        d_all = np.concatenate([dAB, dBA])

        metrics = {
            "name_A": nameA,
            "name_B": nameB,
            "A_to_B_mean":   float(np.mean(dAB)),
            "A_to_B_median": float(np.median(dAB)),
            "A_to_B_max":    float(np.max(dAB)),
            "A_to_B_std":    float(np.std(dAB)),
            "B_to_A_mean":   float(np.mean(dBA)),
            "B_to_A_median": float(np.median(dBA)),
            "B_to_A_max":    float(np.max(dBA)),
            "B_to_A_std":    float(np.std(dBA)),
            "mean":   float(np.mean(d_all)),
            "median": float(np.median(d_all)),
            "rms":    float(np.sqrt(np.mean(d_all ** 2))),
            "hausdorff": float(np.max(d_all)),
        }
        return metrics

    # ------------------------------------------------------------
    # 7. Основной расчёт
    # ------------------------------------------------------------
    def compute(self, s_factor=7.0, k=4, n_fine=5000):
        self.load().compute_centroid().extract_right_contour()

        _, _, x4, y4 = self.build_spline(
            self.x_r, self.y_r, s_factor=s_factor, k=k, n_fine=n_fine
        )
        area4 = self.area_curve_to_chord(x4, y4)
        area_raw = self.area_curve_to_chord(self.x_r, self.y_r)
        diff_area = abs(area4 - area_raw)

        metrics = self.compare_curves(
            x4, y4,
            self.x_r, self.y_r,
            nameA="Сплайн 4",
            nameB="Сырые точки",
        )

        return {
            "diff_area": diff_area,
            "metrics": metrics,
        }


# ------------------------------------------------------------
# Использование
# ------------------------------------------------------------
# if __name__ == "__main__":
#     result = ContourAreaDiff("/content/result.json").compute()
#
#     diff_area = result["diff_area"]
#     metrics = result["metrics"]
#
#     print(f"Разница площадей diff_area = {diff_area:.2f} px²")
#     print(f"объединённое: mean | {metrics['mean']:.3f}")