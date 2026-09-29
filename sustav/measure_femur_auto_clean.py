import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

import re, os

PIXEL_X_MM = 0.60
PIXEL_Y_MM = 1.05

TOP_LIMIT_MM = 30.0
BOTTOM_LIMIT_MM = 30.0
LATERAL_LIMIT_MM = 20.0

BOTTOM_BAND_PX = 8
BRANCH_STOP_Y = 0.15

ISCHIAL_Y_MIN = 0.45
ISCHIAL_Y_MAX = 0.82
ISCHIAL_PROMINENCE = 8.0


def base_name(name):
    return Path(name).stem.split("(")[0]


def load_annotation(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        if shape.get("label") == "sustav":
            return np.asarray(shape["points"], dtype=np.float32)

    raise ValueError("В JSON нет polygon с label='sustav'")


def build_json_map(folder):
    result = {}
    for path in folder.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = Path(data.get("imagePath", "")).name
            if name:
                result.setdefault(name, path)
                result.setdefault(base_name(name), path)
        except (OSError, json.JSONDecodeError):
            continue
    return result


def find_json(image_path, json_map):
    path = json_map.get(image_path.name) or json_map.get(base_name(image_path.name))
    if path is None:
        raise FileNotFoundError(f"JSON не найден для {image_path.name}")
    return path


def get_bottom_endpoints(polygon, h):
    idx = np.flatnonzero(polygon[:, 1] >= h - BOTTOM_BAND_PX)
    if len(idx) < 2:
        raise ValueError("Не найдены нижние точки диафиза")

    return (
        int(idx[np.argmin(polygon[idx, 0])]),
        int(idx[np.argmax(polygon[idx, 0])]),
    )


def walk_branch(polygon, start, step, h):
    points = []
    n = len(polygon)
    i = start

    for _ in range(n):
        p = polygon[i]
        points.append(p)
        if p[1] <= BRANCH_STOP_Y * h:
            break
        i = (i + step) % n

    return np.asarray(points, dtype=np.float32)


def find_ischial(branch, h):
    y = branch[:, 1]
    peaks, props = find_peaks(
        y,
        prominence=ISCHIAL_PROMINENCE,
        distance=2,
    )

    valid = [
        (props["prominences"][i], int(p))
        for i, p in enumerate(peaks)
        if ISCHIAL_Y_MIN * h <= y[p] <= ISCHIAL_Y_MAX * h
    ]

    if not valid:
        return None

    prominence, peak = max(valid, key=lambda x: x[0])
    local = branch[max(0, peak - 3):min(len(branch), peak + 4)]
    max_y = local[:, 1].max()
    local = local[local[:, 1] >= max_y - 1.5]

    return np.mean(local, axis=0), float(prominence)


def find_trochanter(branch, side, h):
    y = branch[:, 1]
    mask = (y >= 0.35 * h) & (y <= 0.75 * h)
    band = branch[mask]
    if len(band) < 3:
        band = branch

    lateral = band[np.argmin(band[:, 0])] if side == "L" else band[np.argmax(band[:, 0])]
    distances = np.sum((branch - lateral) ** 2, axis=1)
    start = int(np.argmin(distances))
    proximal = branch[start:]

    if len(proximal) < 5:
        raise ValueError("Недостаточно точек для поиска большого вертела")

    minima, _ = find_peaks(
        -proximal[:, 1],
        prominence=5.0,
        distance=2,
    )

    if len(minima) == 0:
        raise ValueError("Не найдена верхушка большого вертела")

    return proximal[minima[0]].copy(), lateral.copy()


def detect_points(polygon, h):
    left_idx, right_idx = get_bottom_endpoints(polygon, h)
    left_bottom = polygon[left_idx]
    right_bottom = polygon[right_idx]

    left_branch = walk_branch(polygon, left_idx, -1, h)
    right_branch = walk_branch(polygon, right_idx, 1, h)

    left_ischial = find_ischial(left_branch, h)
    right_ischial = find_ischial(right_branch, h)

    if left_ischial is None and right_ischial is None:
        raise ValueError("Не найдена седалищная точка")

    if left_ischial is None or (
        right_ischial is not None and right_ischial[1] > left_ischial[1]
    ):
        pelvic = right_ischial
        femur_branch = left_branch
    else:
        pelvic = left_ischial
        femur_branch = right_branch

    shaft_x = (left_bottom[0] + right_bottom[0]) / 2.0
    side = "L" if pelvic[0][0] > shaft_x else "R"

    top, lateral = find_trochanter(femur_branch, side, h)

    return {
        "side": side,
        "top": top,
        "lateral": lateral,
        "ischial": pelvic[0],
    }


def measure(points, h, w):
    top_y = float(points["top"][1])
    bottom_y = float(points["ischial"][1])
    lateral_x = float(points["lateral"][0])

    top = top_y * PIXEL_Y_MM
    bottom = (h - 1 - bottom_y) * PIXEL_Y_MM
    lateral = (
        lateral_x * PIXEL_X_MM
        if points["side"] == "L"
        else (w - 1 - lateral_x) * PIXEL_X_MM
    )

    return {
        "top_mm": top,
        "bottom_mm": bottom,
        "lateral_mm": lateral,
    }


def check(d):
    top_ok = d["top_mm"] >= TOP_LIMIT_MM
    bottom_ok = d["bottom_mm"] >= BOTTOM_LIMIT_MM
    lateral_ok = d["lateral_mm"] >= LATERAL_LIMIT_MM
    return top_ok, bottom_ok, lateral_ok, top_ok and bottom_ok and lateral_ok


def visualize(image, polygon, points, d, checks):
    vis = image.copy()
    h, w = vis.shape[:2]

    cv2.polylines(
        vis,
        [np.round(polygon).astype(np.int32)],
        True,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    top_x, top_y = np.round(points["top"]).astype(int)
    lat_x, lat_y = np.round(points["lateral"]).astype(int)
    isc_x, isc_y = np.round(points["ischial"]).astype(int)
    side = points["side"]

    cv2.circle(vis, (top_x, top_y), 4, (0, 0, 255), -1)
    cv2.arrowedLine(vis, (top_x, 0), (top_x, top_y), (0, 255, 255), 2, tipLength=0.04)

    cv2.circle(vis, (isc_x, isc_y), 4, (255, 0, 0), -1)
    cv2.arrowedLine(vis, (isc_x, isc_y), (isc_x, h - 1), (0, 255, 255), 2, tipLength=0.04)

    cv2.circle(vis, (lat_x, lat_y), 4, (255, 0, 255), -1)
    end = (0, lat_y) if side == "L" else (w - 1, lat_y)
    cv2.arrowedLine(vis, (lat_x, lat_y), end, (255, 255, 0), 2, tipLength=0.04)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)

    cv2.putText(vis, f"SIDE: {'LEFT' if side == 'L' else 'RIGHT'}", (5, 20), font, .55, white, 2, cv2.LINE_AA)
    cv2.putText(vis, f"TOP: {d['top_mm']:.1f} mm", (5, 42), font, .55, green if checks[0] else red, 2, cv2.LINE_AA)
    cv2.putText(vis, f"BOTTOM: {d['bottom_mm']:.1f} mm", (5, 64), font, .55, green if checks[1] else red, 2, cv2.LINE_AA)
    cv2.putText(vis, f"LATERAL: {d['lateral_mm']:.1f} mm", (5, 86), font, .55, green if checks[2] else red, 2, cv2.LINE_AA)
    cv2.putText(vis, "CORRECT" if checks[3] else "INCORRECT", (5, h - 12), font, .75, green if checks[3] else red, 2, cv2.LINE_AA)

    return vis

def _split_path(path_image: str) -> tuple[str, str]:
    filename = re.split(r'[\\/]', path_image)[-1]
    study    = re.split(r'[\\/]', path_image)[-2]
    name     = os.path.splitext(filename)[0]
    return name, study

def process(image_path, json_path):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

    h, w = image.shape[:2]
    polygon = load_annotation(json_path)
    points = detect_points(polygon, h)
    distances = measure(points, h, w)
    checks = check(distances)
    name, study = _split_path(image_path)
    out_image = os.path.join("sustavess", f"{study}_{name}.jpg")
    # out_image = (f"{image_path}_result.jpg")
    vis = visualize(image, polygon, points, distances, checks)
    cv2.imwrite(str(out_image), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    report = {
        "side": "left" if points["side"] == "L" else "right",
        "points": {
            "greater_trochanter": [float(x) for x in points["top"]],
            "trochanter_lateral": [float(x) for x in points["lateral"]],
            "ischial": [float(x) for x in points["ischial"]],
        },
        "distances_mm": distances,
        "correct": checks[3]
    }



    return report


def main(image, json):
    report = process(image, json)
    d = report["distances_mm"]
    print(
        f"{'LEFT' if report['side'] == 'left' else 'RIGHT'}, "
        f"TOP={d['top_mm']:.1f} мм, "
        f"BOTTOM={d['bottom_mm']:.1f} мм, "
        f"LATERAL={d['lateral_mm']:.1f} мм -> "
        f"{'CORRECT' if report['correct'] else 'INCORRECT'}"
    )
    return report['correct']



# if __name__ == "__main__":
#     main(
#         "C:\path\to\data.json",
#         "C:\path\to\image.jpg",
#     )
#
