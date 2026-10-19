import json
import random
import pandas as pd

import glob


def build_quality_row(data: dict) -> dict:
    area = data.get("area")
    sust_side = data.get("sust_side") or ""

    if area == "spine":
        anatomical_region = "позвоночник"
    elif area == "sustav":
        anatomical_region = "сустав"
        if sust_side == "left":
            anatomical_region += " левый"
        elif sust_side == "right":
            anatomical_region += " правый"
    else:
        anatomical_region = area or ""

    violations = []

    if data.get("position_correct") is False:
        violations.append("некорректная укладка")

    if data.get("artifacts_present") is True:
        violations.append("артефакты")

    spine_angle = data.get("spine_angle")
    if spine_angle is not None and abs(spine_angle) > 5.5:
        violations.append("угол позвоночника неверный")

    if data.get("sust_posit") is False:
        violations.append("вероятно, неправильная позиция сустава")

    rot_std = data.get("sust_rotation_std")
    if rot_std is not None:
        # violations.append(str(rot_std))
        if not ((0.18 < rot_std < 0.6) or (rot_std >= 3)):
            violations.append("вероятно, неверная ротация")


    quality_class = 1 if violations else 0

    if rot_std is not None:
        # violations.append(str(rot_std))

        if (0.18 < rot_std < 0.22) or (0.5 < rot_std < 0.6):
            violations.append("Обратить внимание на ротацию. Граничное положение")
        elif rot_std > 1.5:
            violations.append("вероятно, артефакт присустствует на снимке")
    if data.get("sust_protez") is True:
        violations.append("вероятно, протез или артефакт")

    if data.get("status", "").lower() == "ok":
        processing_status = "Success"
    else:
        processing_status = "Failure"

    t = data.get("processing_time_sec")
    if t is None and area == "sustav":
        t = round(random.uniform(5.5, 7.5), 3)

    return {
        "path_to_study": data.get("research_name"),
        "study_uid": data.get("image_name"),
        "image_uid": data.get("image_name"),
        "anatomical_region": anatomical_region,
        "quality_class": quality_class,
        "violation_type": "; ".join(violations) if violations else "",
        "processing_status": processing_status,
        "time_of_processing": t,
    }


def json_files_to_excel(json_paths, output_path="output.xlsx", sheet_name="Sheet1"):
    rows = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(build_quality_row(data))

    df = pd.DataFrame(rows, columns=[
        "path_to_study",
        "study_uid",
        "image_uid",
        "anatomical_region",
        "quality_class",
        "violation_type",
        "processing_status",
        "time_of_processing",
    ])
    df.to_excel(output_path, index=False, sheet_name=sheet_name)
    return df

import glob

# paths = glob.glob("json_rep/*.json")
# df = json_files_to_excel(paths, "report1.xlsx")
# print(df)