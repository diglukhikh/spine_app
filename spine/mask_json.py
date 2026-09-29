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
        label: str = "spine",
        input_box: np.ndarray = None,
        input_point: np.ndarray = None,
        input_label: np.ndarray = None,
        epsilon_ratio: float = 0.002,
    ) -> str:

        if input_box is None:
            input_box = np.array([90, 0, 210, 300])
        if input_point is None:
            input_point = np.array([
                (150, 150),
                (150, 200),
                (150, 295),
                (150, 100),
                (250, 50),
                (50, 50),
                (50, 280),
                (250, 280),
            ])
        if input_label is None:
            input_label = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        h, w = image.shape[:2]

        self.predictor.set_image(image)

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
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


# if __name__ == "__main__":
#     CHECKPOINT = r"C:\MobileSAM-master\weights\mobile_sam.pt"
#     IMAGE_PATH = r"CR015864.jpg"
#     OUTPUT_JSON = r"CR015864.json"
#
#     segmenter = SpineSegmenter(CHECKPOINT)
#     segmenter.process(IMAGE_PATH, OUTPUT_JSON)