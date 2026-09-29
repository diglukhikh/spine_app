import cv2
import json
import os
import numpy as np


class ObjectRemover:
    def __init__(self, image_path: str, json_path: str, output_path,
                 output_dir: str = None, expand_percent: float = 30):
        """
        :param image_path: путь к исходному изображению
        :param json_path:  путь к JSON с разметкой (label == "spine")
        :param output_dir: куда сохранять; по умолчанию — рядом с исходником
        :param expand_percent: на сколько процентов расширить контур по X
        """
        self.image_path = os.path.abspath(image_path)
        self.json_path = os.path.abspath(json_path)
        self.expand_percent = expand_percent

        if output_dir is None:
            output_dir = os.path.dirname(self.image_path) or os.getcwd()
        self.output_dir = os.path.abspath(output_dir)

        self.img = None
        self.H = self.W = None
        self.contour = None
        self.contour_expanded = None
        self.mask_inside = None
        self.mask_outside = None
        self.result_filled = None
        self.output_path = output_path

    # ---------- расширение контура ----------
    @staticmethod
    def _expand_contour_percent(contour, percent):
        pts = contour.reshape(-1, 2).astype(np.float64)
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        cx = x + w / 2.0
        k = 1.0 + percent / 100.0
        pts_scaled = pts.copy()
        pts_scaled[:, 0] = (pts[:, 0] - cx) * k + cx
        return pts_scaled.round().astype(np.int32)

    # ---------- загрузка ----------
    def _load(self):
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Нет файла изображения: {self.image_path}")
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"Нет файла JSON: {self.json_path}")

        # читаем через numpy + imdecode — работает с любыми путями (кириллица, пробелы)
        with open(self.image_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cv2.imdecode не смог прочитать: {self.image_path}")

        self.img = img
        self.H, self.W = img.shape

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shape = next((s for s in data.get("shapes", []) if s.get("label") == "spine"), None)
        if shape is None:
            raise ValueError("В JSON нет shape с label='spine'")
        self.contour = np.array(shape["points"], dtype=np.int32)

    # ---------- маски ----------
    def _build_masks(self):
        self.contour_expanded = self._expand_contour_percent(
            self.contour, self.expand_percent
        )
        self.mask_inside = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(self.mask_inside, [self.contour_expanded], 255)
        self.mask_outside = cv2.bitwise_not(self.mask_inside)

    # ---------- обработка ----------
    def _process(self):
        outside_vals = self.img[self.mask_outside == 255]
        fill_value = int(np.median(outside_vals)) if outside_vals.size else 0

        self.result_filled = self.img.copy()
        self.result_filled[self.mask_inside == 255] = fill_value

    # ---------- сохранение через os ----------
    def _save(self):
        # создаём папку, если её нет
        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        file_name = f"{base_name}_without_object_filled.png"
        self.output_path = os.path.join(self.output_dir, file_name)

        # кодируем в память
        ok, buf = cv2.imencode(".png", self.result_filled)
        if not ok:
            raise IOError("cv2.imencode не смог закодировать PNG")

        # пишем файл через os (надёжно на любых путях)
        try:
            with open(self.output_path, "wb") as f:
                f.write(buf.tobytes())
        except OSError as e:
            raise IOError(f"Не удалось записать {self.output_path}: {e}") from e

        # проверяем, что файл реально появился и не пустой
        if not os.path.exists(self.output_path):
            raise IOError(f"Файл не создан: {self.output_path}")
        size = os.path.getsize(self.output_path)
        if size == 0:
            raise IOError(f"Файл пустой: {self.output_path}")

        print(f"[ObjectRemover] Сохранено: {self.output_path} ({size} байт)")
        return self.output_path

    # ---------- публичный запуск ----------
    def run(self):
        self._load()
        self._build_masks()
        self._process()
        return self._save()


# ---------- пример использования ----------
# if __name__ == "__main__":
#     remover = ObjectRemover(
#         image_path="/content/52532.jpg",
#         json_path="/content/52532.json",
#         output_dir="/content",       # или любая папка внутри Flask-проекта
#         expand_percent=30,
#     )
#     out = remover.run()
#     print("Итог:", out)