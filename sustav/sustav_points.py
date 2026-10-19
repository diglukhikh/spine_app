import numpy as np
import cv2


def get_three_points(image_path: str) -> np.ndarray:
    """
    Находит 3 точки на суставе, отталкиваясь от центра масс яркой области.
    Возвращает np.ndarray формы (3, 2) с координатами (x, y).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть: {image_path}")

    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        raise ValueError("Объект не найден")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    joint_mask = (labels == largest).astype(np.uint8)

    # Центр масс
    M = cv2.moments(joint_mask)
    if M["m00"] == 0:
        raise ValueError("Пустая маска")
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Главная ось через PCA
    ys, xs = np.nonzero(joint_mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts_centered = pts - [cx, cy]

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    major_axis /= np.linalg.norm(major_axis)

    proj = pts_centered @ major_axis
    half_len = np.percentile(np.abs(proj), 90)

    # Три точки: центр масс + два конца вдоль главной оси
    raw_points = np.array([
        [cx, cy],
        [cx + major_axis[0] * half_len * 0.7, cy + major_axis[1] * half_len * 0.7],
        [cx - major_axis[0] * half_len * 0.7, cy - major_axis[1] * half_len * 0.7],
    ])

    # Округляем и приводим к int
    points = np.round(raw_points).astype(int)

    # Притягиваем точки внутрь маски, если выпали
    h, w = joint_mask.shape
    mask_pts = np.column_stack([xs, ys])
    for i, (x, y) in enumerate(points):
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if joint_mask[y, x] == 0:
            d = (mask_pts[:, 0] - x) ** 2 + (mask_pts[:, 1] - y) ** 2
            nx, ny = mask_pts[int(np.argmin(d))]
            points[i] = (int(nx), int(ny))
        else:
            points[i] = (x, y)

    return points


if __name__ == "__main__":
    path = "CR000005.jpg"
    points = get_three_points(path)
    print("Три точки для промпта сегментации:", points)

    # Визуализация
    img = cv2.imread(path)
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    cv2.imwrite("points_vis.jpg", img)