import cv2
import numpy as np
import os


class PelvisDetector:
    # ================= вспомогательные функции =================

    @staticmethod
    def find_spine_columns(bright_local: np.ndarray, bw: int,
                           search_frac: float = 0.5,
                           peak_ratio: float = 0.30,
                           pad_frac: float = 0.02) -> tuple:
        """
        Границы полосы позвоночника по столбцовой проекции.
        Широкая полоса: столбцы, где проекция >= peak_ratio * пик.
        """
        col = bright_local.mean(axis=0)
        cx = bw // 2
        half_search = int(bw * search_frac / 2)
        x_lo = max(0, cx - half_search)
        x_hi = min(bw, cx + half_search)

        window = col[x_lo:x_hi]
        if window.size == 0 or window.max() <= 0:
            half = int(bw * 0.12 / 2)
            return cx - half, cx + half

        peak = window.max()
        thr = peak * peak_ratio
        idx = np.where(window >= thr)[0]
        x0 = x_lo + int(idx.min())
        x1 = x_lo + int(idx.max()) + 1

        pad = max(3, int(bw * pad_frac))
        return max(0, x0 - pad), min(bw, x1 + pad)

    @staticmethod
    def is_pelvis_like(blob: dict, bh: int, bw: int,
                       spine_x0: int, spine_x1: int) -> tuple:
        """Фильтр формы. Возвращает (ok, reason)."""
        x, y, w, h = blob["x"], blob["y"], blob["w"], blob["h"]
        area = blob["area"]

        elongation = max(w, h) / (min(w, h) + 1e-6)
        if elongation < 1.3:
            return False, "not_elongated"

        blob_cx = x + w / 2
        if spine_x0 < blob_cx < spine_x1:
            return False, "inside_spine"
        if blob_cx < bw * 0.03 or blob_cx > bw * 0.97:
            return False, "at_edge"

        if y + h < bh * 0.25:
            return False, "too_high"

        fill_ratio = area / (w * h + 1e-6)
        if fill_ratio > 0.92:
            return False, "too_solid"

        return True, "ok"

    @staticmethod
    def near_spine_x(blob: dict, spine_x0: int, spine_x1: int,
                     bw: int, max_dx_frac: float) -> tuple:
        """Блоб должен быть сбоку от оси позвоночника в пределах max_dx."""
        bx = blob["x"] + blob["w"] / 2
        dx = max(0, spine_x0 - bx, bx - spine_x1)
        if dx > bw * max_dx_frac:
            return False, "too_far_from_spine"
        return True, "ok"

    @staticmethod
    def classify_blobs(blobs, bh, bw, spine_x0, spine_x1,
                       min_blob_contrast, min_blob_area_frac,
                       weak_contrast_ratio, weak_area_ratio,
                       spine_dx_frac, total_area):
        """
        Разбивает блобы на strong / weak.
        Форма и связь с осью — общие.
        Отличаются пороги контраста и площади.
        """
        weak_min_contrast = min_blob_contrast * weak_contrast_ratio
        weak_min_area_frac = min_blob_area_frac * weak_area_ratio

        strong, weak = [], []
        for b in blobs:
            ok, reason = PelvisDetector.is_pelvis_like(
                b, bh, bw, spine_x0, spine_x1)
            if not ok:
                b["reject"] = reason
                continue
            ok, reason = PelvisDetector.near_spine_x(
                b, spine_x0, spine_x1, bw, spine_dx_frac)
            if not ok:
                b["reject"] = reason
                continue

            area_frac = b["area"] / total_area
            if (b["mean_contrast"] >= min_blob_contrast
                    and area_frac >= min_blob_area_frac):
                b["level"] = "strong"
                strong.append(b)
            elif (b["mean_contrast"] >= weak_min_contrast
                    and area_frac >= weak_min_area_frac):
                b["level"] = "weak"
                weak.append(b)
            else:
                b["reject"] = "below_weak_threshold"

        return strong, weak

    @staticmethod
    def check_pair_asymmetric(strong, weak, spine_x0, spine_x1,
                              bh, pair_dy_frac):
        """
        Асимметричная парность:
          1) strong + strong
          2) strong + weak (в любую сторону)
        Возвращает (has_pair, pair, pair_dy, pair_type).
        pair_type: 'strong_strong' | 'strong_weak' | None
        """
        def side(b):
            cx = b["x"] + b["w"] / 2
            if cx < spine_x0:
                return "L"
            if cx > spine_x1:
                return "R"
            return None

        strong_L = [b for b in strong if side(b) == "L"]
        strong_R = [b for b in strong if side(b) == "R"]
        weak_L = [b for b in weak if side(b) == "L"]
        weak_R = [b for b in weak if side(b) == "R"]

        max_dy = bh * pair_dy_frac

        def best_pair(list_a, list_b):
            best, best_dy = None, None
            for a in list_a:
                acy = a["y"] + a["h"] / 2
                for b in list_b:
                    bcy = b["y"] + b["h"] / 2
                    dy = abs(acy - bcy)
                    if dy <= max_dy and (best_dy is None or dy < best_dy):
                        best, best_dy = (a, b), dy
            return best, best_dy

        pair, dy = best_pair(strong_L, strong_R)
        if pair is not None:
            return True, pair, dy, "strong_strong"

        pair, dy = best_pair(strong_L, weak_R)
        if pair is not None:
            return True, pair, dy, "strong_weak"

        pair, dy = best_pair(strong_R, weak_L)
        if pair is not None:
            return True, pair, dy, "strong_weak"

        return False, None, None, None

    # ================= отрисовка =================

    @staticmethod
    def _expand_rect(x, y, w, h,
                     target_scale: float = 3.0,
                     max_w: int = 70,
                     max_h: int = 50):
        """
        Расширяет прямоугольник от центра в target_scale раз,
        но не более max_w по ширине и max_h по высоте.
        """
        cx = x + w / 2.0
        cy = y + h / 2.0

        new_w = min(w * target_scale, max_w)
        new_h = min(h * target_scale, max_h)

        nx = int(round(cx - new_w / 2.0))
        ny = int(round(cy - new_h / 2.0))
        return nx, ny, int(round(new_w)), int(round(new_h))

    @staticmethod
    def _draw_pair_vis(bottom_gray: np.ndarray, pair) -> np.ndarray:
        """
        Возвращает BGR-изображение с красными расширенными прямоугольниками
        только для финальной пары.
        """
        vis = cv2.cvtColor(bottom_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if pair is not None:
            for b in pair:
                x, y, w_, h_ = b["x"], b["y"], b["w"], b["h"]
                nx, ny, nw, nh = PelvisDetector._expand_rect(
                    x, y, w_, h_,
                    target_scale=3.0,
                    max_w=70,
                    max_h=50,
                )
                cv2.rectangle(vis, (nx, ny), (nx + nw, ny + nh),
                              (0, 0, 255), 2)
        return vis

    # ================= основная функция =================

    def detect_pelvis(
        self,
        image_path: str,
        bottom_frac: float = 0.40,
        spine_search_frac: float = 0.50,
        spine_peak_ratio: float = 0.30,
        local_win: int = 31,
        contrast_percentile: float = 60.0,
        min_positive_frac: float = 0.003,
        min_blob_area_frac: float = 0.0004,
        min_blob_contrast: float = 0.006,
        weak_contrast_ratio: float = 0.5,
        weak_area_ratio: float = 0.5,
        spine_dx_frac: float = 0.30,
        pair_dy_frac: float = 0.45,
    ) -> dict:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img)

        y0 = int(h * (1 - bottom_frac))
        bottom = img_eq[y0:, :].astype(np.float32)
        bh, bw = bottom.shape

        # 1. Локальный контраст
        if local_win % 2 == 0:
            local_win += 1
        local_mean = cv2.blur(bottom, (local_win, local_win))
        local_contrast = bottom - local_mean

        positive = local_contrast[local_contrast > 0]
        if positive.size == 0:
            return {"has_pelvis": False, "reason": "no_positive_contrast",
                    "n_blobs_raw": 0, "n_strong": 0, "n_weak": 0,
                    "positive_frac": 0.0}

        thr = np.percentile(positive, contrast_percentile)
        bright_local = (local_contrast > thr).astype(np.uint8)
        bright_local = cv2.morphologyEx(bright_local, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
        bright_local = cv2.morphologyEx(bright_local, cv2.MORPH_CLOSE,
                                        np.ones((5, 5), np.uint8))

        # 2. Позвоночник
        spine_x0, spine_x1 = self.find_spine_columns(
            bright_local, bw,
            search_frac=spine_search_frac,
            peak_ratio=spine_peak_ratio,
        )

        # 3. Маска вне позвоночника
        body_mask = np.ones_like(bright_local)
        body_mask[:, spine_x0:spine_x1] = 0
        bright_body = bright_local * body_mask
        positive_frac = bright_body.sum() / (body_mask.sum() + 1e-6)

        # 4. Компоненты
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_body, connectivity=8)
        total_area = bh * bw
        blobs = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area / total_area < min_blob_area_frac * weak_area_ratio:
                continue
            comp = (labels == i)
            blobs.append({
                "area": int(area),
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "mean_contrast": float(local_contrast[comp].mean()),
                "mean_bright": float(bottom[comp].mean()),
            })

        # 5. Классификация strong / weak
        strong, weak = self.classify_blobs(
            blobs, bh, bw, spine_x0, spine_x1,
            min_blob_contrast, min_blob_area_frac,
            weak_contrast_ratio, weak_area_ratio,
            spine_dx_frac, total_area,
        )

        # 6. Асимметричная парность
        has_pair, pair, pair_dy, pair_type = self.check_pair_asymmetric(
            strong, weak, spine_x0, spine_x1, bh, pair_dy_frac
        )

        has_pelvis = has_pair and positive_frac >= min_positive_frac

        result = {
            "has_pelvis": bool(has_pelvis),
            "has_pair": bool(has_pair),
            "pair_type": pair_type,
            "pair_dy": None if pair_dy is None else float(pair_dy),
            "positive_frac": float(positive_frac),
            "n_blobs_raw": len(blobs),
            "n_strong": len(strong),
            "n_weak": len(weak),
            "spine_x0": int(spine_x0),
            "spine_x1": int(spine_x1),
            "strong_blobs": strong,
            "weak_blobs": weak,
            "all_blobs": blobs,
        }

        # ===== Сохранение _vis.jpg =====
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(os.path.abspath(image_path))
        vis_path = os.path.join(out_dir, f"{base}_vis.jpg")

        vis = self._draw_pair_vis(bottom, pair)
        cv2.imwrite(vis_path, vis)
        result["vis_path"] = vis_path

        return result


# ================= УПРОЩЁННЫЙ ЗАПУСК =================

def detect_pelvis(image_path: str) -> dict:
    """
    Упрощённый интерфейс: одна картинка -> result + путь к _vis.jpg.
    require_pair=True, allow_single=False всегда.
    """
    return PelvisDetector().detect_pelvis(image_path)


if __name__ == "__main__":
    import json

    image_path = r"CR019080.jpg"   # <-- поменяйте на свой путь

    result = detect_pelvis(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))