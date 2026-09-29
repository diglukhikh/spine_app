import os
from pathlib import Path
from PIL import Image
import imagehash


class ImageDuplicateRemover:
    """Находит и удаляет дубликаты изображений в папке."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def __init__(self, folder, threshold=5):
        self.folder = Path(folder)
        self.threshold = threshold
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Папка не найдена: {self.folder}")

    def _list_files(self):
        """Возвращает список файлов-изображений в папке."""
        return [
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in self.EXTENSIONS
        ]

    @staticmethod
    def _group_by_size(files):
        """Группирует файлы по размеру (быстрая отсечка)."""
        by_size = {}
        for f in files:
            size = f.stat().st_size
            by_size.setdefault(size, []).append(f)
        return by_size

    def _perceptual_equal(self, p1, p2):
        """Сравнение изображений через перцептивный хеш."""
        try:
            h1 = imagehash.phash(Image.open(p1))
            h2 = imagehash.phash(Image.open(p2))
            return h1 - h2 <= self.threshold
        except Exception as e:
            print(f"  ⚠ Ошибка сравнения {p1.name} и {p2.name}: {e}")
            return False

    def find_duplicates(self):
        """Возвращает список групп дубликатов: [[оригинал, дубль1, ...], ...]."""
        files = self._list_files()
        print(f"Найдено файлов: {len(files)}\n")

        by_size = self._group_by_size(files)
        duplicates = []

        for group in by_size.values():
            if len(group) < 2:
                continue

            used = set()
            for i, f1 in enumerate(group):
                if f1 in used:
                    continue
                dupes = [f1]
                for f2 in group[i + 1:]:
                    if f2 in used:
                        continue
                    if self._perceptual_equal(f1, f2):
                        dupes.append(f2)
                        used.add(f2)
                if len(dupes) > 1:
                    used.add(f1)
                    duplicates.append(dupes)

        return duplicates

    def remove_duplicates(self):
        """Удаляет дубликаты, оставляя по одному файлу из каждой группы."""
        dups = self.find_duplicates()

        if not dups:
            print("Дубликатов не найдено.")
            return 0

        total_removed = 0
        for group in dups:
            keep = group[0]
            print(f"Оригинал: {keep.name}")
            for d in group[1:]:
                print(f"  → дубликат: {d.name}")
                try:
                    d.unlink()
                    total_removed += 1
                except Exception as e:
                    print(f"    ⚠ Не удалось удалить: {e}")

        print(f"\nВсего групп дубликатов: {len(dups)}")
        print(f"Удалено файлов: {total_removed}")
        return total_removed


if __name__ == "__main__":
    FOLDER = r"upload_dicom/2.25.105033841136787477881326326053319223817"
    remover = ImageDuplicateRemover(FOLDER)
    remover.remove_duplicates()