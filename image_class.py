import cv2
import h5py
import numpy as np
import tensorflow as tf
from collections import Counter
import os

try:
    from PIL import Image, ImageOps
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


IMAGE_EXTS = ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.tif', '.tiff', '.webp')


class AreaClassifier:
    """
    Классификатор фрагментов изображения (JPG/PNG/...).

    Параметры
    ----------
    db_path : str
        Путь к HDF5-базе эталонов.
    target : int
        Размер, к которому приводится изображение/фрагмент (должен совпадать
        с тем, при котором собиралась база — иначе размерность не сойдётся).
    k : int
        Число ближайших соседей.
    apply_exif : bool
        Применять ли EXIF-ориентацию (актуально для JPG с телефонов).
    """

    def __init__(self, db_path: str, target: int = 256, k: int = 1,
                 apply_exif: bool = True):
        self.db_path = db_path
        self.target = target
        self.k = k
        self.apply_exif = apply_exif

        self.X_ref = None
        self.y_ref = None
        self.names_ref = None

        self.load_reference_db_h5()

    # ========================================================
    # 1. Загрузка базы из .h5
    # ========================================================
    def load_reference_db_h5(self, path: str = None):
        if path is None:
            path = self.db_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f'HDF5-база не найдена: {path}')

        with h5py.File(path, 'r') as f:
            self.X_ref = f['X'][:].astype(np.float32)
            self.y_ref = np.array([s.decode() if isinstance(s, bytes) else str(s)
                                   for s in f['y'][:]])
            self.names_ref = [s.decode() if isinstance(s, bytes) else str(s)
                              for s in f['names'][:]]
        return self.X_ref, self.y_ref, self.names_ref

    # ========================================================
    # 2. Чтение JPG (и других форматов) -> grayscale np.ndarray
    # ========================================================
    @staticmethod
    def _read_image(path, apply_exif: bool = True):
        path = str(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Файл не найден: {path}')

        # --- вариант с EXIF (для JPG с телефона) ---
        if apply_exif and _PIL_OK:
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    im = im.convert('L')
                    img = np.array(im, dtype=np.uint8)
                if img is not None and img.size > 0:
                    return img
            except Exception:
                pass

        # --- запасной вариант: OpenCV ---
        flags = cv2.IMREAD_GRAYSCALE
        if hasattr(cv2, 'IMREAD_IGNORE_ORIENTATION'):
            flags |= cv2.IMREAD_IGNORE_ORIENTATION
        img = cv2.imread(path, flags)
        if img is None:
            raise FileNotFoundError(f'Не читается изображение: {path}')
        return img

    # ========================================================
    # 3. Картинка -> вектор (ВАРИАНТ B: ровно 5 pooling'ов)
    # ========================================================
    def _array_to_vector(self, img, target: int = None):
        """Принимает grayscale np.ndarray, возвращает вектор длины 128."""
        if target is None:
            target = self.target

        if img is None:
            raise ValueError('Пустой массив изображения')

        # если пришёл цветной — переводим в grayscale
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)

        filters = [
            img,
            cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)),
        ]
        stack = np.stack([f.astype(np.float32) for f in filters], axis=-1) / 255.0

        x = tf.constant(stack)[None, ...]  # (1, target, target, 2)

        # --- ровно 5 pooling'ов, как в исходной версии ---
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')  # 256 -> 128
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')  # 128 -> 64
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')  # 64  -> 32
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')  # 32  -> 16
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')  # 16  -> 8
        # итог: 8 * 8 * 2 = 128

        return tf.reshape(x, [-1]).numpy()

    def image_to_vector(self, path, target: int = None):
        """Читает файл (JPG/PNG/...) и возвращает вектор."""
        img = self._read_image(path, apply_exif=self.apply_exif)
        return self._array_to_vector(img, target=target)

    def _to_vector(self, image, target: int = None):
        """Универсальный вход: путь или np.ndarray."""
        if isinstance(image, (str, bytes, os.PathLike)):
            return self.image_to_vector(image, target=target)
        if isinstance(image, np.ndarray):
            return self._array_to_vector(image, target=target)
        raise TypeError(
            f'Ожидался путь или np.ndarray, получено: {type(image)}'
        )

    # ========================================================
    # 4. Классификация вектора
    # ========================================================
    def _classify_vector(self, v, k: int = None, return_details: bool = False):
        if k is None:
            k = self.k

        d = np.linalg.norm(self.X_ref - v[None, :], axis=1)

        k = min(k, len(d))
        nn_idx = np.argsort(d)[:k]
        nn_labels = self.y_ref[nn_idx]
        nn_dists = d[nn_idx]

        label = Counter(nn_labels).most_common(1)[0][0]

        if not return_details:
            return label

        weights = 1.0 / (nn_dists + 1e-8)
        score = {}
        for lab in set(self.y_ref):
            mask = nn_labels == lab
            score[lab] = float(weights[mask].sum() / weights.sum())

        return label, {
            'score': score,
            'neighbors': list(zip(nn_labels.tolist(), nn_dists.tolist())),
            'min_dist': float(d.min()),
            'mean_dist': float(d.mean()),
        }

    # ========================================================
    # 5. Публичные методы классификации
    # ========================================================
    def classify_image(self, image, k: int = None, return_details: bool = False):
        """Классифицирует изображение (JPG/PNG/... или np.ndarray)."""
        v = self._to_vector(image)
        return self._classify_vector(v, k=k, return_details=return_details)

    def classify(self, image, k: int = None, return_details: bool = False):
        """Алиас для обратной совместимости."""
        return self.classify_image(image, k=k, return_details=return_details)

    def classify_many(self, images, k: int = None, return_details: bool = False):
        """Пакетная классификация списка путей/массивов."""
        return [self.classify_image(img, k=k, return_details=return_details)
                for img in images]


# ============================================================
# Пример под JPG
# ============================================================
if __name__ == '__main__':
    clf = AreaClassifier('class/part.h5', target=256, k=3)

    # # проверка размерности
    # v = clf.image_to_vector('photo.jpg')
    # print('feat_dim из базы:    ', clf.X_ref.shape[1])   # 128
    # print('feat_dim из картинки:', v.size)               # 128

    # классификация JPG с диска
    label = clf.classify_image("upload_dicom/upload_20260927_222413/CR000001.jpg")
    print('label:', label)
    #
    # # с подробностями
    # label, details = clf.classify_image('photo.jpg', return_details=True)
    # print('label:', label)
    # print('score:', details['score'])
    #
    # # пакет JPG
    # results = clf.classify_many(['a.jpg', 'b.jpeg', 'c.JPG'])
    # print(results)