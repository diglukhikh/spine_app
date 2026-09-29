import json
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor


class SpineSegmenter:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = device
        self.sam = sam_model_registry["vit_t"](checkpoint=checkpoint).to(device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.002):
        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        eps = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        return points if len(points) >= 3 else None

    def process(
        self,
        image_path: str,
        output_json: str,
        label: str = "sustav",
        input_box: np.ndarray = None,
        input_point: np.ndarray = None,
        input_label: np.ndarray = None,
        epsilon_ratio: float = 0.002,
    ) -> str:
        """
        Сегментация изображения с помощью MobileSAM.

        Параметры
        ----------
        image_path : str
            Путь к входному изображению.
        output_json : str
            Путь для сохранения результата в формате LabelMe.
        label : str
            Название класса для shape.
        input_box : np.ndarray, optional
            Bounding box в формате [x1, y1, x2, y2].
        input_point : np.ndarray, optional
            Точки-подсказки в формате Nx2.
        input_label : np.ndarray, optional
            Метки для точек (1 — foreground, 0 — background).
        epsilon_ratio : float
            Коэффициент аппроксимации контура.
        """
        if input_point is None and input_box is None:
            raise ValueError(
                "Необходимо передать хотя бы один из аргументов: "
                "input_point или input_box."
            )

        # Если точки переданы, но метки — нет, считаем все точки foreground.
        if input_point is not None and input_label is None:
            input_label = np.ones(len(input_point), dtype=np.int32)

        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # if gray is None:
        #     raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        #
        # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        h, w = image.shape[:2]

        self.predictor.set_image(image)

        # Формируем аргументы для предиктора: box должен быть либо None,
        # либо массивом формы (1, 4).
        box_arg = input_box[None, :] if input_box is not None else None

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box_arg,
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]

        points = self._mask_to_polygon(best_mask, epsilon_ratio)
        if points is None:
            raise RuntimeError("Не удалось построить полигон из маски")

        labelme_json = {
            "version": "0.3.3",
            "flags": {},
            "shapes": [
                {
                    "label": label,
                    "text": "",
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            "imagePath": image_path.split("\\")[-1].split("/")[-1],
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)

        return output_json


# segmenter = SpineSegmenter(checkpoint="mobile_sam.pt", device="cpu")
#
# input_point = np.array([
#     (150, 150),
#     (150, 200),
#     (150, 295),
#     (150, 100),
#     (250, 50),
#     (50, 50),
#     (50, 280),
#     (250, 280),
# ])
# input_label = np.array([1, 1, 1, 1, 0, 0, 0, 0])
#
# segmenter.process(
#     image_path="image.png",
#     output_json="result.json",
#     input_point=input_point,
#     input_label=input_label,
# )