import os
import pydicom
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

def apply_window(img, center, width):
    img = img.astype(np.float32)
    low = center - width / 2
    high = center + width / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low) * 255.0
    return img.astype(np.uint8)

def dicom_to_jpeg(dcm_path, jpg_path, sharpness=2.8, quality=95):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    # Приводим к 8 битам
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-9) * 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L")



    # Резкость — после масштабирования
    if sharpness != 1.0:
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness)

    # Создаём папку для файла, если её нет
    os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
    if os.path.isdir(jpg_path) or not os.path.splitext(jpg_path)[1]:
        os.makedirs(jpg_path, exist_ok=True)
        base = os.path.splitext(os.path.basename(dcm_path))[0]
        jpg_path = os.path.join(jpg_path, base + ".jpg")

    pil_img.save(jpg_path, "JPEG", quality=quality)


    # 5. Сохранение
    # sharpened_img.save(jpg_path, "JPEG", quality=100)



# def to_jpg(adress):
#     root = Path(adress)
#     out_folder = Path("upload_dicom") / root.name
#
#     dcm_files = list(root.rglob("*.dcm"))
#     print(f"Найдено DICOM-файлов: {len(dcm_files)}")
#
#     out_folder.mkdir(parents=True, exist_ok=True)
#
#     for dcm_path in dcm_files:
#         out_path = out_folder / (dcm_path.stem + ".jpg")
#         dicom_to_jpeg(str(dcm_path), str(out_path))
#         print("OK:", dcm_path, "->", out_path)

def to_jpg(adress, out_path):
    out_path = out_path / (adress.stem + ".jpg")
    dicom_to_jpeg(adress, out_path)
    print("OK", adress)
    return


# dicom_to_jpeg(r"C:\Users\Хозяин\PycharmProjects\bones_hack\2.25.78999454687125053757525542837745968726\series_004_3_CR\A2504222617 DXA\CR DXA\CR000001.dcm", r"C:\Users\Хозяин\PycharmProjects\bones_hack\test_save", sharpness=2.8)


# root = Path(r"C:\Users\Хозяин\PycharmProjects\bones_hack\2.25.78999454687125053757525542837745968726")
# # out_folder = Path("upload_dicom") / root.name
#
#
#
#
# if __name__ == "__main__":
#     to_jpg(root, "test")