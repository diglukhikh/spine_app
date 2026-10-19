from sustav import place, segment, measure_femur_auto_clean, rotation_new, sustav_points, find_artefacts_sustav
import numpy as np
import os
import re
from PIL import Image

import json



CHECKPOINT      = os.path.join("sam", "mobile_sam.pt")


JSON_DIR        = "rotation_chek"


def _split_path(path_image: str) -> tuple[str, str]:
    filename = re.split(r'[\\/]', path_image)[-1]
    study    = re.split(r'[\\/]', path_image)[-2]
    name     = os.path.splitext(filename)[0]
    return name, study


def save_json(original_path, points, out_path):
    with open(original_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['shapes'][0]['points'] = points.tolist()

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Поиск точек в пустоте
def points(image_path):
    detector = place.BlackPointsDetector(image_path, threshold=30,
                                   max_dist=70, n=3, x_tolerance=2)

    points = detector.get_points()
    # 2. Новые Y-координаты
    new_points = detector.get_new_y_points()

    # 3. Формируем массив вида [(x, new_y), ...]
    result = np.array([(p['x'], int(round(p['y']))) for p in new_points])

    result1 = np.array([(detector.cx, detector.cy)])

    print(result1)

    return result, result1

# сегментация пустоты
def segment_void(image_path, coords):
    segmenter = segment.SustavSegmenter(CHECKPOINT, device="cpu")

    input_point = coords
    input_label = np.array([1, 1, 1])
    # name, study = _split_path(path)
    output_json = os.path.join(JSON_DIR, "result.json")
    segmenter.process(
        image_path=image_path,
        output_json=output_json,
        input_point=input_point,
        input_label=input_label,
    )

    return output_json

# сегментация сустава
def segment_sust(image_path, coords):
    segmenter = segment.SustavSegmenter(CHECKPOINT, device="cpu")
    input_point = coords
    input_label = np.array([1, 1, 1])
    # name, study = _split_path(path)
    output_json = os.path.join(JSON_DIR, "result_sust.json")
    segmenter.process(
        image_path=image_path,
        output_json=output_json,
        input_point=input_point,
        input_label=input_label,
    )
    return output_json

# проверка стороны сустава

def side_check(image_path, output_json_void, centr_void):
    with open(output_json_void, 'r', encoding='utf-8') as f:
        data = json.load(f)
    points = np.asarray(data['shapes'][0]['points'], dtype=float)
    output_json = os.path.join(JSON_DIR, "result.json")
    mirrored = points.copy()
    img = Image.open(image_path)
    width, _ = img.size
    side = "left"
    print("value", centr_void[0][0])
    if centr_void[0][0] > width / 2:
        mirrored[:, 0] = width - points[:, 0]
        side = "right"
        save_json(output_json_void, mirrored, output_json)
        return side
    return side



def main(path):
    side, bumps_max, rasst, protez = None,None,None,False
    enum_pics = 5


    #область интересов на основе яркости пикселей
    coords_void, coord = points(path)

    print(coords_void, type(coords_void))
    try:
        #сегментация пустоты для оценки ротации
        output_json_void = segment_void(path, coords_void)

        #проверка стороны (json приводится к левой стороне для оценки ротации)
        side = side_check(path, output_json_void, coord)
        print(side)
        # проверка ротации
        finder = rotation_new.ContourBumpFinder(output_json_void, threshold=0.002)
        bumps, peaks = finder.find_bumps()

        # сохранение визуализации ротации
        path1 = finder.save_plot()
        print("Сохранено:", path1)

        print(bumps)

        top3 = sorted(bumps, key=lambda p: p[2], reverse=True)[:3]
        bumps_max = sum(p[2] for p in top3) / len(top3)

        # bumps_max = max(bumps, key=lambda p: p[2])[2]

        enum_pics = sum(1 for p in bumps if p[2] > 0.8)

    except:
        pass

    detector1 = find_artefacts_sustav.BrightAreaDetector(threshold=252, min_area=350)
    artefacts1 = detector1.process(path)

    detector2 = find_artefacts_sustav.BrightAreaDetector(threshold=240, min_area=800)
    artefacts2 = detector2.process(path)

    try:
        if artefacts1 or artefacts2 and enum_pics > 3:
            protez = True
    except:
        if artefacts1 or artefacts2:
            protez = True



    #промпты для сустава
    try:
        points_sust = sustav_points.get_three_points(path)
        print(points_sust, type(points_sust))
        #сегментация сустава
        output_json_sust = segment_sust(path, points_sust)

        rasst = measure_femur_auto_clean.main(path, output_json_sust)
        print(rasst)
    except:
        rasst = False
        pass

    return side, bumps_max, rasst, protez


# print(main(path))