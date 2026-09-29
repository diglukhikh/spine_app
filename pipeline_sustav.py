from sustav import place, segment, rotation, measure_femur_auto_clean, rotation_new
import numpy as np
import os
import re


CHECKPOINT      = os.path.join("sam", "mobile_sam.pt")
JSON_DIR        = "jsonmy"

def points(path):
    detector = place.BlackPointsDetector(path, threshold=30,
                                   max_dist=70, n=3, x_tolerance=2)

    points = detector.get_points()
    print(detector.cx)
    print(f"Центр масс: cx={detector.cx:.2f}, cy={detector.cy:.2f}")
    print(f"Найдено {len(points)} чёрных точек:")
    for i, p in enumerate(points, 1):
        print(f"  {i}) x={p['x']}, y={p['y']}, яркость={p['intensity']}")

    # 2. Новые Y-координаты
    new_points = detector.get_new_y_points()

    # 3. Формируем массив вида [(x, new_y), ...]
    result = np.array([(p['x'], int(round(p['y']))) for p in new_points])

    result1 = np.array([(detector.cx, detector.cy)])

    print("\nРезультат (x, new_y):")
    print(result)

    return result, result1


from PIL import Image, ImageOps

def _split_path(path_image: str) -> tuple[str, str]:
    filename = re.split(r'[\\/]', path_image)[-1]
    study    = re.split(r'[\\/]', path_image)[-2]
    name     = os.path.splitext(filename)[0]
    return name, study

def mirror_if_left_inplace(image_path: str, value):
    """
    Отзеркаливает изображение по адресу, если value < половины ширины.
    Перезаписывает исходный файл.
    """
    value = value[0]
    img = Image.open(image_path)
    width, _ = img.size
    side = "left"
    print("value", value)
    if value > width / 2:
        img = ImageOps.mirror(img)
        img.save(image_path)
        side = "right"
        return side
    return side



# Использование





def segment_bone(path, coords):
    segmenter = segment.SpineSegmenter(checkpoint=CHECKPOINT, device="cpu")

    input_point = coords
    input_label = np.array([1])
    name, study = _split_path(path)
    output_json = os.path.join(JSON_DIR, f"{study}_{name}.json")

    segmenter.process(
        image_path=path,
        output_json=output_json,
        input_point=input_point,
        input_label=input_label,
    )

    return output_json


# def segment_void(path, coords):
#     segmenter = segment.SpineSegmenter(checkpoint=CHECKPOINT, device="cpu")
#
#     input_point = coords
#     input_label = np.array([1, 1, 1])
#     name, study = _split_path(path)
#     output_json = os.path.join(JSON_DIR, f"{study}_{name}.json")
#     try:
#         segmenter.process(
#             image_path=path,
#             output_json=output_json,
#             input_point=input_point,
#             input_label=input_label,
#         )
#     except:
#         input_label = np.array([1])
#         segmenter.process(
#             image_path=path,
#             output_json=output_json,
#             input_point=input_point,
#             input_label=input_label,
#         )
#     finally:
#         pass
#
#
#     return output_json
    
    
def rot(path_json):

    diff_area = rotation.ContourAreaDiff(path_json).compute()
    criteria = diff_area["metrics"]
    print(criteria["mean"])

    return criteria





# Пример использования

def main(path):
    criteria, side, rasst = None,None,None
    try:
        points2,x_c = points(path)

        print(x_c[0])
    except:
        pass

    try:
        side = mirror_if_left_inplace(path, value=x_c[0])
    except:
        pass

    try:
        points1, x_c = points(path)
    except:
        pass

    try:
        json_path = segment_bone(path, x_c)
    except:
        pass

    # json_path = segment_void(path, points1)
    try:
        rasst = measure_femur_auto_clean.main(path, json_path)
        print(rasst)
    except:
        pass

    try:
        criteria = rotation_new.analyze_hip_rotation(json_path)
    except:
        pass


    return criteria, side, rasst

# main(path)