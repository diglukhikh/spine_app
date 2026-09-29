import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor

CHECKPOINT = r"C:\MobileSAM-master\weights\mobile_sam.pt"

device = "cpu"
sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT).to(device)
sam.eval()
predictor = SamPredictor(sam)

# --- ЧБ изображение ---
gray = cv2.imread(r"CR000002.jpg", cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError("Не удалось прочитать изображение")

image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

predictor.set_image(image)

# --- Точки ---
# label = 1: внутри объекта (кость)
# label = 0: вне объекта (фон)
input_box = np.array([110, 0, 200, 300])

input_point = np.array([
    (150, 150),   # 1 — внутри нужной структуры
    (150, 200),
    (150, 295),
    (150, 10),
    (250, 50),
    (50, 50),
    (50, 280),
    (250, 280)
])
input_label = np.array([1, 1, 1 , 1, 0,0 ,0,0])

masks, scores, _ = predictor.predict(
    # point_coords=input_point,
    # point_labels=input_label,
    box=input_box[None, :],
    multimask_output=True,
)
best_mask = masks[np.argmax(scores)]

# --- Сохраняем результат ---
overlay = image.copy()
overlay[best_mask] = [255, 0, 0]
result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

# # Отметим точки для наглядности
for (x, y), lbl in zip(input_point, input_label):
    color = (0, 255, 0) if lbl == 1 else (0, 0, 255)   # зелёный / красный
    cv2.circle(result, (int(x), int(y)), 5, color, -1)

cv2.imwrite("result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print(f"Лучший score: {max(scores):.3f}")