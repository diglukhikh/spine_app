import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без интерактивного окна
import matplotlib.pyplot as plt
from matplotlib.path import Path
import chardet
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

class ContourBumpFinder:
    """
    Находит точки выпуклостей на правой границе контура из JSON (формат labelme).

    Использование:
        finder = ContourBumpFinder("/path/to/CR000001.json")
        bumps = finder.find_bumps()          # список точек (x, y)
        finder.save_plot("sustav")           # сохранит sustav/CR000001.png
    """

    def __init__(self, json_path, threshold=0.03, min_distance=1,
                 smooth_sigma=1, spline_s=1.1, trim_frac=0.12,
                 output_dir="rotation_chek"):
        self.json_path = json_path
        self.threshold = threshold
        self.min_distance = min_distance
        self.smooth_sigma = smooth_sigma
        self.spline_s = spline_s
        self.trim_frac = trim_frac
        self.output_dir = output_dir

        # результаты, заполняются по мере работы
        self.data = None
        self.poly_path = None
        self.H = None
        self.W = None
        self.mean = None
        self.point_min_y = None
        self.right_pts = None
        self.x_trim = None
        self.y_trim = None
        self.curvature = None
        self.x_s = None
        self.y_s = None
        self.peaks = None

    # ------------------------------------------------------------------ #
    #  1. Загрузка JSON
    # ------------------------------------------------------------------ #
    def load(self):
        with open(self.json_path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        text = raw.decode(detected["encoding"])
        self.data = json.loads(text)

        shape = self.data["shapes"][0]
        pts = np.array(shape["points"], dtype=float)
        self.H = self.data["imageHeight"]
        self.W = self.data["imageWidth"]
        self.poly_path = Path(pts)
        self.pts = pts

    # ------------------------------------------------------------------ #
    #  2. Центр масс
    # ------------------------------------------------------------------ #
    def compute_centroid(self):
        H, W = self.H, self.W
        yy, xx = np.mgrid[0:H, 0:W]
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = self.poly_path.contains_points(grid_pts).reshape(H, W)
        ys, xs = np.where(inside)
        pts_inside = np.column_stack([xs, ys]).astype(float)
        self.mean = pts_inside.mean(axis=0)

    # ------------------------------------------------------------------ #
    #  3. Правая граница: фильтрация и сортировка
    # ------------------------------------------------------------------ #
    def extract_right_boundary(self):
        pts = self.pts
        mean = self.mean

        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        y_cut = y_max - 0.05 * (y_max - y_min)
        pts = pts[pts[:, 1] <= y_cut]

        right_mask = pts[:, 0] >= mean[0]
        right_pts = pts[right_mask]

        order = np.argsort(right_pts[:, 1])
        right_pts = right_pts[order]

        if len(right_pts) == 0:
            self.point_min_y = None
            self.right_pts = right_pts
            return

        idx_min_y = np.argmin(right_pts[:, 1])
        self.point_min_y = right_pts[idx_min_y]

        keep_mask = right_pts[:, 0] >= self.point_min_y[0]
        right_pts = right_pts[keep_mask]
        self.right_pts = right_pts

    def _drop_step_segments(self, x, y, min_len_frac=0.2, max_len_frac=5.0,
                            angle_thr_deg=35.0, kappa_mad_k=3.0):
        """
        Отбраковка «ступенек»: коротких сегментов с большим поворотом.
        Возвращает очищенные x, y (без изменения формы кривой).
        """
        if len(x) < 3:
            return x, y

        pts = np.column_stack([x, y])
        v = np.diff(pts, axis=0)  # (N-1, 2)
        L = np.linalg.norm(v, axis=1)  # длины сегментов
        L_med = np.median(L)

        # 1) длина сегмента в разумных пределах (отсекает «микрошаги»)
        seg_ok = (L >= min_len_frac * L_med) & (L <= max_len_frac * L_med)

        # 2) угол поворота на сегмент, нормированный на длину
        if len(v) >= 2:
            v1, v2 = v[:-1], v[1:]
            cos = (v1 * v2).sum(1) / (
                    np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
            )
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            kappa_seg = angle / np.maximum(L[:-1], 1e-9)

            k_med = np.median(kappa_seg)
            k_mad = 1.4826 * np.median(np.abs(kappa_seg - k_med))
            k_thr = k_med + kappa_mad_k * max(k_mad, 1e-9)

            # 3) сегмент плох, если короткий ИЛИ (угол большой И κ_seg > порога)
            seg_ok[:-1] &= ~((angle > angle_thr_deg) & (kappa_seg > k_thr))

        # переводим «сегменты» в «точки»: точка i хороша, если оба смежных сегмента ок
        keep = np.ones(len(x), dtype=bool)
        keep[:-1] &= seg_ok
        keep[1:] &= seg_ok
        # концы не трогаем — иначе сломается trim по краям
        keep[0] = keep[-1] = True

        return x[keep], y[keep]
    # ------------------------------------------------------------------ #
    #  4. Аппроксимация сплайном + обрезка краёв
    # ------------------------------------------------------------------ #
    def fit_spline(self):
        right_pts = self.right_pts
        x_r = right_pts[:, 0]
        y_r = right_pts[:, 1]

        # ======================= ОТБРАКОВКА «СТУПЕНЕК» =======================
        x_r, y_r = self._drop_step_segments(x_r, y_r)
        if len(x_r) < 2:
            # защита: если всё выкинули — откатываемся
            right_pts = self.right_pts
            x_r, y_r = right_pts[:, 0], right_pts[:, 1]
        # ====================================================================

        # --- параметризация по длине дуги ---
        t = np.zeros(len(x_r))
        t[1:] = np.cumsum(np.hypot(np.diff(x_r), np.diff(y_r)))
        t /= t[-1]

        # ======================= НОВОЕ: фильтр выбросов кривизны =======================
        n = len(x_r)
        keep = np.ones(n, dtype=bool)

        # 1) лёгкое сглаживание координат, чтобы κ не шумела (окно нечётное, >=3)
        win = min(5, n if n % 2 == 1 else n - 1)
        if win >= 3 and n >= 5:
            xs = savgol_filter(x_r, win, 2)
            ys = savgol_filter(y_r, win, 2)
        else:
            xs, ys = x_r, y_r

        # 2) параметрическая кривизна по сглаженным координатам
        dx, dy = np.gradient(xs), np.gradient(ys)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = (dx * dx + dy * dy) ** 1.5
        denom[denom == 0] = np.nan
        kappa = np.abs(dx * ddy - dy * ddx) / denom



        # 3) робастная отбраковка по MAD (устойчиво к самому выбросу)
        k_clean = kappa[np.isfinite(kappa)]
        if len(k_clean) >= 3:
            med = np.median(k_clean)
            mad = 1.4826 * np.median(np.abs(k_clean - med))
            mad_k = getattr(self, "curv_mad_k", 3.0)  # можно настраивать
            if mad > 0:
                keep = np.abs(kappa - med) <= mad_k * mad
                keep[~np.isfinite(kappa)] = True  # NaN не выкидываем
            else:
                keep[:] = True
        # всегда оставляем концы
        keep[0] = keep[-1] = True

        # 4) если после фильтра осталось слишком мало точек — откатываемся
        if keep.sum() < 4:
            keep[:] = True

        x_f, y_f, t_f = x_r[keep], y_r[keep], t[keep]
        # перенормируем параметр под новую выборку (splprep требует возрастающий u)
        t_f = (t_f - t_f[0]) / (t_f[-1] - t_f[0]) if t_f[-1] > t_f[0] else t_f
        # ==============================================================================

        # --- s ---
        s_val = 0.0 if getattr(self, "spline_s", 0) == 0 else len(x_f) * self.spline_s * 1e-3

        k = min(1, len(x_f) - 1)

        spl, u = splprep([x_f, y_f], u=t_f, s=s_val, k=k)

        u_fine = np.linspace(0, 1, 300)
        x_fine, y_fine = splev(u_fine, spl)

        n_fine = len(x_fine)
        k_trim_start = int(n_fine * (self.trim_frac - 0.05))
        k_trim = int(n_fine * (self.trim_frac + 0.08))

        # print("kappa :", np.round(kappa, 3))
        # print("keep  :", keep.astype(int))
        # print("med,mad:", med, mad, "threshold:", mad_k * mad)
        # print("left:", x_r[keep], y_r[keep])

        self.x_trim = x_fine[k_trim_start: n_fine - k_trim]
        self.y_trim = y_fine[k_trim_start: n_fine - k_trim]

        # print("right_pts.shape =", self.right_pts.shape)
        # print("right_pts =", self.right_pts)
        # print("t =", t)

    # ------------------------------------------------------------------ #
    #  5. Кривизна и поиск пиков
    # ------------------------------------------------------------------ #
    def compute_curvature(self):
        x, y = self.x_trim, self.y_trim
        x_s = gaussian_filter1d(x, sigma=self.smooth_sigma)
        y_s = gaussian_filter1d(y, sigma=self.smooth_sigma)

        dx = np.gradient(x_s)
        dy = np.gradient(y_s)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

        self.x_s = x_s
        self.y_s = y_s
        self.curvature = curvature

    def find_bumps(self):
        self.load()
        self.compute_centroid()
        self.extract_right_boundary()
        self.fit_spline()
        self.compute_curvature()

        curvature = self.curvature
        threshold = self.threshold
        if threshold is None:
            threshold = np.mean(curvature) + np.std(curvature)

        peaks, _ = find_peaks(curvature, height=threshold,
                              distance=self.min_distance)
        self.peaks = peaks

        bumps = [(float(self.x_s[p]), float(self.y_s[p]), float(self.curvature[p])) for p in peaks]
        return bumps, peaks

    # ------------------------------------------------------------------ #
    #  6. Сохранение графика (без интерактива, перевёрнут по Y)
    # ------------------------------------------------------------------ #
    def save_plot(self, output_dir=None, filename=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            base = os.path.splitext(os.path.basename(self.json_path))[0]
            filename = base + ".png"

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(self.x_trim, self.y_trim, 'b-', lw=1.5,
                alpha=0.4, label='Сплайн')
        ax.plot(self.x_s, self.y_s, 'b-', lw=2, label='Сглаженная линия')

        if self.peaks is not None and len(self.peaks) > 0:
            ax.plot(self.x_s[self.peaks], self.y_s[self.peaks], 'ro',
                    ms=10, markerfacecolor='none', mew=2,
                    label='Выпуклости')
            for i, p in enumerate(self.peaks):
                ax.annotate(
                    f'#{i+1}\n({self.x_s[p]:.1f}, {self.y_s[p]:.1f})',
                    xy=(self.x_s[p], self.y_s[p]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Линия с найденными выпуклостями',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # перевёрнутая ось Y
        ax.set_ylim(self.H, 0)
        ax.set_xlim(0, self.W)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


# ---------------------------------------------------------------------- #
#  Пример использования
# ---------------------------------------------------------------------- #
# if __name__ == "__main__":
#     finder = ContourBumpFinder("result.json", threshold=0.03)
#     bumps, peaks = finder.find_bumps()
#     print(f"Найдено выпуклостей: {len(bumps)}")
#     # for i, (x, y) in enumerate(bumps, 1):
#     #     print(f"  #{i}: ({x:.1f}, {y:.1f}, {y:.1f})")
#     path = finder.save_plot()
#     print("Сохранено:", path)
#     print(bumps)
