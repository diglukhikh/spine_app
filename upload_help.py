from pathlib import Path
import pydicom
import numpy as np
import os
import zipfile
from PIL import Image, ImageEnhance

import shutil

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)



BASE_DIR = Path(__file__).resolve().parent
UP_DIR   = BASE_DIR / "up_dicom"
UP_DIR_JPG   = BASE_DIR / "upload_dicom"
UP_DIR.mkdir(exist_ok=True)


def clear_up_dir_jpg():
    """
    Полностью очищает папку UP_DIR_JPG:
    удаляет все файлы и подпапки внутри неё,
    но саму папку UP_DIR_JPG оставляет.
    """
    up_dir = Path(UP_DIR_JPG)

    if not up_dir.exists():
        return {"deleted": 0, "errors": []}

    deleted = 0
    errors = []

    for item in up_dir.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=False)
            else:
                item.unlink()
            deleted += 1
        except Exception as e:
            errors.append(f"{item.name}: {e}")

    return {"deleted": deleted, "errors": errors}

def apply_window(img, center, width):
    img = img.astype(np.float32)
    low = center - width / 2
    high = center + width / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low) * 255.0
    return img.astype(np.uint8)


def dicom_to_jpeg(dcm_path, jpg_path, sharpness_factor=3.0):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width  = getattr(ds, "WindowWidth", None)

    if center is not None and width is not None:
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        img8 = apply_window(img, float(center), float(width))
    else:
        img8 = ((img - img.min()) / (img.max() - img.min() + 1e-9) * 255).astype(np.uint8)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img8 = 255 - img8

    pil_img = Image.fromarray(img8, mode="L")
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharpened_img = enhancer.enhance(sharpness_factor)

    # гарантируем, что папка под jpg существует
    Path(jpg_path).parent.mkdir(parents=True, exist_ok=True)
    sharpened_img.save(jpg_path, "JPEG", quality=95)


# ---------- Совмещённая обработка папки ----------
def process_folder(folder: Path, out_root: Path):
    """
    Обходит folder, каждый DICOM конвертирует в JPG.
    Структура схлопывается до одного уровня: берётся первая
    подпапка (например, UID исследования), остальные уровни
    отбрасываются. Имя JPG = имя DICOM со сменой расширения.
    """
    folder   = Path(folder)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(folder):
        root_path = Path(root)
        rel_dir   = root_path.relative_to(folder)

        # Берём только первый сегмент относительного пути
        if rel_dir.parts:
            target_dir = out_root / rel_dir.parts[0]
        else:
            target_dir = out_root
        target_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            fpath    = root_path / fname
            jpg_path = target_dir / (fpath.stem + ".jpg")
            try:
                dicom_to_jpeg(str(fpath), str(jpg_path), sharpness_factor=3.0)
            except Exception as e:
                print(f"dicom_to_jpeg failed for {fpath}: {e!r}")


def _sanitize_rel_path(raw: str) -> Path:
    """
    Превращает 'patient_01/series_2/IMG001.dcm' в безопасный Path.
    Защита от ../ и абсолютных путей, отбрасываем опасные компоненты.
    """
    raw = raw.replace("\\", "/").lstrip("/")
    parts = []
    for part in raw.split("/"):
        if not part or part in (".", ".."):
            continue
        cleaned = "".join(c for c in part if c.isalnum() or c in "._- ")
        if cleaned:
            parts.append(cleaned)
    if not parts:
        parts = ["unnamed"]
    return Path(*parts)

def _unique_path(path: Path) -> Path:
    """
    Если файл по указанному пути существует — возвращает свободный путь
    с суффиксом _1, _2, _3, ... до первого свободного.
    Если файла нет — возвращает исходный путь без изменений.
    """
    if not path.exists():
        return path

    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def process_zip(zip_path: Path, extract_to: Path, out_root: Path):
    """
    Распаковывает ZIP в extract_to, затем прогоняет содержимое через
    process_folder.

    Правила распределения JPG:
      - несколько корневых директориев → каждый идёт в out_root/<имя_директория>/...
      - один корневой директорий        → он же в out_root/<имя_директория>/...
      - плоский архив (файлы в корне)   → в out_root/<имя_архива>/...

    То есть имя архива используется как fallback-контейнер только когда
    структурировать по папкам невозможно.
    """
    zip_path   = Path(zip_path)
    extract_to = Path(extract_to)
    out_root   = Path(out_root)

    extract_to.mkdir(parents=False, exist_ok=True)
    out_root.mkdir(parents=False, exist_ok=True)


    with zipfile.ZipFile(zip_path) as zf:
        # защита от zip-slip
        base = extract_to.resolve()
        for member in zf.namelist():
            target = (extract_to / member).resolve()
            if not str(target).startswith(str(base)):
                raise ValueError(f"Недопустимый путь в архиве: {member}")
        zf.extractall(extract_to)

    root = _descend_single_wrapper(extract_to)

    entries    = list(root.iterdir())
    subdirs    = [p for p in entries if p.is_dir()]
    root_files = [p for p in entries if p.is_file()]

    # Случай 1: в корне есть файлы (плоский архив или смесь).
    # Структуры нет — используем имя архива как папку.
    if root_files and not subdirs:
        archive_dir = out_root / _safe_stem(zip_path.stem)
        archive_dir.mkdir(parents=False, exist_ok=True)
        process_folder(root, archive_dir)
        return

    # Случай 2: смесь — есть и файлы, и папки в корне.
    # Файлы из корня кладём в out_root/<имя_архива>/, папки — как обычно.
    if root_files and subdirs:
        archive_dir = out_root / _safe_stem(zip_path.stem)
        archive_dir.mkdir(parents=True, exist_ok=True)
        # файлы из корня
        for f in root_files:
            jpg_path = _unique_path(archive_dir / (f.stem + ".jpg"))
            try:
                dicom_to_jpeg(str(f), str(jpg_path), sharpness_factor=3.0)
            except Exception as e:
                print(f"dicom_to_jpeg failed for {f}: {e!r}")
        # папки
        for d in subdirs:
            target = out_root / _safe_stem(d.name)
            target.mkdir(parents=True, exist_ok=True)
            process_folder(d, target)
        return

    # Случай 3: только директории в корне.
    # Каждый корневой директорий → отдельная папка в out_root.
    # print(subdirs)
    for d in subdirs:
        print(d)
        target = out_root / _safe_stem(d.name)
        target.mkdir(parents=False, exist_ok=True)
        process_folder(d, target)

def _descend_single_wrapper(folder: Path) -> Path:
    """
    Если внутри folder ровно одна подпапка и нет файлов —
    спускаемся в неё и повторяем, пока не упрёмся в уровень,
    где больше одной подпапки или есть файлы.
    """
    folder = Path(folder)
    while True:
        entries = list(folder.iterdir())
        subdirs = [p for p in entries if p.is_dir()]
        files   = [p for p in entries if p.is_file()]

        if len(subdirs) == 1 and not files:
            folder = subdirs[0]
            continue
        return folder

def _safe_stem(stem: str) -> str:
    """Чистим имя архива от мусора, чтобы получить безопасное имя папки."""
    cleaned = "".join(c for c in stem if c.isalnum() or c in "._- ").strip()
    return cleaned or "archive"


def count_files_in_dir(root_dir) -> int:
    """Считает общее количество файлов во всех папках (рекурсивно) внутри root_dir."""
    total = 0
    if not os.path.isdir(root_dir):
        return 0
    for _, _, files in os.walk(root_dir):
        total += len(files)
    return total



