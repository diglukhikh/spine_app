import cv2
import h5py
import numpy as np
import tensorflow as tf
from collections import Counter
import os


class FragmentClassifier:
    """
    Классификатор фрагментов изображения.

    Параметры
    ----------
    db_path : str
        Путь к HDF5-базе эталонов.
    target : int
        Размер, к которому приводится изображение/фрагмент.
    k : int
        Число ближайших соседей.
    """

    def __init__(self, db_path: str, target: int = 64, k: int = 1):
        self.db_path = db_path
        self.target = target
        self.k = k

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
        with h5py.File(path, 'r') as f:
            self.X_ref = f['X'][:].astype(np.float32)
            self.y_ref = np.array([s.decode() for s in f['y'][:]])
            self.names_ref = [s.decode() for s in f['names'][:]]
        return self.X_ref, self.y_ref, self.names_ref

    # 2. Картинка -> вектор

    def _array_to_vector(self, img, target: int = None):
        """Принимает grayscale np.ndarray, возвращает вектор."""
        if target is None:
            target = self.target

        if img is None:
            raise ValueError('Пустой массив изображения')

        # если пришёл цветной — переводим в grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)

        filters = [
            img,
            cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F)),
            cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)),
            cv2.Canny(img, 100, 200),
        ]
        stack = np.stack([f.astype(np.float32) for f in filters], axis=-1) / 255.0

        x = tf.constant(stack)[None, ...]
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')

        return tf.reshape(x, [-1]).numpy()

    def image_to_vector(self, path, target: int = None):
        """Читает файл и возвращает вектор."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Не читается: {path}')
        return self._array_to_vector(img, target=target)

    # ========================================================
    # 3. Вырезка фрагментов по bbox
    # ========================================================
    def extract_fragments(self, image_path: str, bboxes):
        """
        Вырезает фрагменты из изображения.

        Параметры
        ----------
        image_path : str
            Путь к исходному изображению.
        bboxes : list of tuple
            Список координат в формате (x1, y1, x2, y2).

        Возвращает
        ----------
        fragments : list of np.ndarray
            Список вырезанных фрагментов (BGR).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f'Не читается: {image_path}')

        h, w = img.shape[:2]
        fragments = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                fragments.append(None)
                continue

            fragments.append(img[y1:y2, x1:x2].copy())

        return fragments

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
    # 5. Классификация изображения
    # ========================================================
    def classify(self, path, k: int = None, return_details: bool = False):
        v = self.image_to_vector(path)
        return self._classify_vector(v, k=k, return_details=return_details)

    # ========================================================
    # ========================================================
    # 6. Классификация фрагментов изображения по bbox
    # ========================================================
    def classify_fragments(self, image_path: str, bboxes,
                           k: int = None, return_details: bool = False):
        fragments = self.extract_fragments(image_path, bboxes)
        results = []

        resultss = []

        for frag in fragments:
            if frag is None:
                results.append(False)
                continue

            v = self._array_to_vector(frag)
            out = self._classify_vector(v, k=k, return_details=return_details)

            label = out[0] if return_details else out
            resultss.append(bool(label))

        for bbox, frag in zip(bboxes, fragments):
            if frag is None:
                results.append({
                    'bbox': tuple(bbox),
                    'label': None,
                    'error': 'invalid bbox',
                })
                continue

            # напрямую через массив — без записи на диск
            v = self._array_to_vector(frag)
            out = self._classify_vector(v, k=k, return_details=return_details)

            if return_details:
                label, info = out
                results.append({
                    'bbox': tuple(bbox),
                    'label': label,
                    'info': info,
                })
            else:
                results.append({
                    'bbox': tuple(bbox),
                    'label': out,
                })

        return results, resultss
    # ========================================================
    # ========================================================
    # 7. Визуализация фрагментов с классами
    # ========================================================
    def visualize_fragments(self, image_path: str, bboxes,
                            results=None, k: int = None,
                            figsize_per_row: float = 3.0,
                            show_full_image: bool = True,
                            only_true: bool = True,
                            true_label: str = 'true',
                            save_path: str = None):
        """
        Визуализирует исходное изображение с bbox и вырезанные фрагменты
        с подписанными классами.

        Параметры
        ----------
        image_path : str
            Путь к исходному изображению.
        bboxes : list of tuple
            Координаты фрагментов (x1, y1, x2, y2).
        results : list of dict, optional
            Результат classify_fragments. Если None — будет вычислен.
        k : int, optional
            Число соседей для классификации (если results=None).
        figsize_per_row : float
            Ширина одной ячейки на графике.
        show_full_image : bool
            Показывать ли исходное изображение с bbox.
        only_true : bool
            Визуализировать только фрагменты, классифицированные как true_label.
        true_label : str
            Имя "истинного" класса.
        save_path : str, optional
            Если задан — сохранит итоговую фигуру и (для исходного изображения)
            картинку с нарисованными рамками по этому пути.
            Для фигуры добавляется суффикс '_grid.png' (если нет расширения),
            для картинки с рамками — '_boxes.png'.
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if results is None:
            results = self.classify_fragments(image_path, bboxes, k=k,
                                              return_details=False)

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f'Не читается: {image_path}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if save_path is not None:
            out_dir = os.path.dirname(os.path.abspath(save_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        # --- фильтрация: индексы фрагментов, которые нужно показать ---
        if only_true:
            keep_idx = [
                i for i, r in enumerate(results)
                if r.get('label') == true_label
            ]
        else:
            keep_idx = list(range(len(bboxes)))

        # ====================================================
        # Сохранение изображения с рамками только по true-объектам
        # ====================================================
        if save_path is not None:
            boxed = img_rgb.copy()
            for i in keep_idx:
                x1, y1, x2, y2 = map(int, bboxes[i])
                color = (0, 255, 0)  # зелёный для true
                cv2.rectangle(boxed, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    boxed, f'{i}:{results[i].get("label")}',
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            base, ext = os.path.splitext(save_path)
            boxes_path = f'{base}_boxes{ext or ".png"}'
            cv2.imwrite(boxes_path, cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
            print(f'Сохранено изображение с рамками: {boxes_path}')

        # ====================================================
        # Сетка визуализации
        # ====================================================
        n = len(keep_idx)
        if n == 0:
            print(f'Нет фрагментов с классом "{true_label}" — визуализировать нечего.')
            return None

        ncols = min(n, 5)
        nrows = int(np.ceil(n / ncols))

        total_rows = nrows + (1 if show_full_image else 0)
        fig, axes = plt.subplots(
            total_rows, ncols,
            figsize=(figsize_per_row * ncols, figsize_per_row * total_rows),
            squeeze=False,
        )

        row_offset = 0

        # --- исходное изображение с bbox (только для true) ---
        if show_full_image:
            ax_full = axes[0, 0]
            ax_full.imshow(img_rgb)
            ax_full.set_title('Исходное изображение', fontsize=10)
            ax_full.axis('off')
            for i in keep_idx:
                x1, y1, x2, y2 = bboxes[i]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime', facecolor='none',
                )
                ax_full.add_patch(rect)
                ax_full.text(x1, y1 - 3, str(i), color='lime',
                             fontsize=9, weight='bold')
            for j in range(1, ncols):
                axes[0, j].axis('off')
            row_offset = 1

        # --- фрагменты ---
        fragments = self.extract_fragments(image_path, bboxes)

        for pos, i in enumerate(keep_idx):
            frag = fragments[i]
            r = pos // ncols
            c = pos % ncols
            ax = axes[r + row_offset, c]

            if frag is None:
                ax.axis('off')
                ax.set_title(f'#{i}  невалидный bbox', fontsize=9)
                continue

            frag_rgb = cv2.cvtColor(frag, cv2.COLOR_BGR2RGB)
            ax.imshow(frag_rgb)
            ax.axis('off')

            label = results[i].get('label')
            title = f'#{i}  {label}' if label is not None else f'#{i}'
            ax.set_title(title, fontsize=10)

            if 'info' in results[i]:
                info = results[i]['info']
                score = info.get('score', {})
                score_str = ', '.join(f'{k_}:{v:.2f}' for k_, v in score.items())
                ax.set_xlabel(f'd={info["min_dist"]:.3f}\n{score_str}',
                              fontsize=8)

        # скрыть пустые ячейки
        for i in range(n, nrows * ncols):
            r = i // ncols
            c = i % ncols
            axes[r + row_offset, c].axis('off')

        # plt.tight_layout()

        # --- сохранение фигуры ---
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            grid_path = f'{base}_grid{ext or ".png"}'
            fig.savefig(grid_path, dpi=150, bbox_inches='tight')
            print(f'Сохранена сетка визуализации: {grid_path}')

        # plt.show()
        return fig

