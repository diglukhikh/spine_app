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

def dicom_to_jpeg(dcm_path, jpg_path, sharpness_factor=3.0):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    # 1. Применяем Rescale Slope/Intercept (Modality LUT)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    # 2. Применяем Window Center/Width (VOI LUT)
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)

    if center is not None and width is not None:
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        img8 = apply_window(img, float(center), float(width))
    else:
        # Если окна нет — делаем простую нормализацию (менее качественно)
        img8 = ((img - img.min()) / (img.max() - img.min() + 1e-9) * 255).astype(np.uint8)

    # 3. Инверсия для MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img8 = 255 - img8

    # 4. Конвертация в Image и повышение резкости
    pil_img = Image.fromarray(img8, mode="L")
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharpened_img = enhancer.enhance(sharpness_factor)

    # 5. Сохранение
    sharpened_img.save(jpg_path, "JPEG", quality=95)



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






# if __name__ == "__main__":
#     to_jpg("2.25.102755089973625799055786462646268820450")