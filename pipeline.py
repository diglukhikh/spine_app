from image_class import AreaClassifier
from spine.spine_estimator import SpineAngleEstimator
from spine.spine_delete import ObjectRemover
from spine.proposals import FragmentExtractor
from spine.siamesse_check import FragmentClassifier
from spine.mask_json import SpineSegmenter
from spine.pelvis_v2_clean import detect_pelvis
from spine.vertabrae_new import SpineAnalyzer
from spine.position_check import SpineAligner
from find_copy import ImageDuplicateRemover

import shutil
from pathlib import Path
import os
import re
import time
import json
import traceback
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, Any
import pipeline_sustav


CHECKPOINT      = os.path.join("sam", "mobile_sam.pt")
CLASSIFIER_PATH = os.path.join("class", "part.h5")
SPINE_PT        = os.path.join("spine", "artefacts.h5")
RESULT_DIR      = "result"
OPERATION_DIR   = "operation"
JSON_DIR        = "jsonmy"
JSON_DIR_report = "json_rep"
OUTPUT_DIR      = os.path.join(os.getcwd(), "output")



@dataclass
class StudyReport:
    """Отчёт по исследованию позвоночника."""
    # --- основные поля ---
    research_name: str = ""
    image_name: str = ""
    area: str = "spine"
    position_correct: Optional[bool] = None
    center_offset: Optional[bool] = None
    artifacts_present: str = ""
    spine_angle: Optional[float] = None
    sust_posit: str = ""
    sust_rotation_std: str = ""
    sust_side: str = ""

    taz: str = ""
    pozv: str = ""

    # --- служебные поля ---
    processing_time_sec: Optional[float] = None   # общее время обработки
    steps_time_sec: dict = field(default_factory=dict)   # время по шагам
    errors: list = field(default_factory=list)    # список ошибок
    status: str = "pending"                       # pending / ok / partial / failed

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    def print_summary(self) -> None:
        print(json.dumps(self.to_dict(), ensure_ascii=False, indent=4))

    def add_error(self, step: str, exc: Exception) -> None:
        self.errors.append({
            "step": step,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })

def safe_step(step_name: str):
    """
    Оборачивает шаг пайплайна:
      - замеряет время выполнения;
      - ловит исключения и пишет их в report.errors;
      - не даёт упасть всему пайплайну.
    Функция-шаг должна первым аргументом принимать StudyReport.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(report: StudyReport, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(report, *args, **kwargs)
                return result
            except Exception as exc:
                report.add_error(step_name, exc)
                print(f"[{step_name}] ОШИБКА: {type(exc).__name__}: {exc}")
                return None
            finally:
                dt = time.perf_counter() - t0
                report.steps_time_sec[step_name] = round(dt, 3)
                print(f"[{step_name}] время: {dt:.3f} с")
        return wrapper
    return decorator


CLEAN_DIRS = [OPERATION_DIR, JSON_DIR]


def clear_dirs(dirs=None, recreate=True):
    if dirs is None:
        dirs = CLEAN_DIRS

    for d in dirs:
        p = Path(d)
        if p.exists():
            for entry in p.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        elif recreate:
            p.mkdir(parents=True, exist_ok=True)


def _split_path(path_image: str) -> tuple[str, str]:
    filename = re.split(r'[\\/]', path_image)[-1]
    study    = re.split(r'[\\/]', path_image)[-2]
    name     = os.path.splitext(filename)[0]
    return name, study


@safe_step("classify")
def step_classify(report: StudyReport, path_image: str) -> Optional[str]:
    """Шаг 1. Классификация снимка."""
    clf = AreaClassifier(CLASSIFIER_PATH, target=256, k=3)
    label = clf.classify_image(path_image)
    print(f"[classify] label = {label}")
    return label


@safe_step("check_centering")
def step_check_centering(report: StudyReport, path_image: str) -> None:
    """Шаг 2. Проверка центрирования позвоночника."""
    aligner = SpineAligner(
        threshold_percentile=70,
        center_zone=0.10,
        method="mass",
    )
    info = aligner.process(path_image)
    report.center_offset = info["centered"]

    if not info["ok"]:
        raise RuntimeError(f"aligner: {info['reason']}")
    elif info["centered"]:
        print("[align] Позвоночник по центру")
    else:
        print("[align] Позвоночник смещён — изображение отцентрировано")


@safe_step("segmentation")
def step_segmentation(report: StudyReport, path_image: str) -> Optional[str]:
    """Шаг 3. Сегментация позвоночника."""
    name, study = _split_path(path_image)
    report.image_name    = os.path.basename(path_image)

    path = Path(path_image)
    folder = path.parts[1]
    report.research_name = folder

    output_json = os.path.join(JSON_DIR, f"{study}_{name}.json")
    segmenter = SpineSegmenter(CHECKPOINT)
    segmenter.process(path_image, output_json)
    print("[segment] JSON сохранён")
    return output_json


@safe_step("estimate_angle")
def step_estimate_angle(report: StudyReport, path_image: str, json_path):
    """Шаг 4. Оценка угла позвоночника."""
    _, study = _split_path(path_image)
    name, study = _split_path(path_image)
    save_dir = RESULT_DIR

    path = Path(path_image)
    folder = path.parts[1]

    save_file = os.path.join(save_dir, f"{folder}_{name}_angle.png")
    estimator = SpineAngleEstimator(
        path_image,
        json_path, save_file)
    angle = estimator.estimate()


    angle = estimator.estimate()
    report.spine_angle = angle
    print(f"[angle] {angle}")
    return angle


@safe_step("remove_spine")
def step_remove_spine(report: StudyReport, path_image: str, output_json: str) -> Optional[str]:
    """Шаг 5. Удаление позвоночника."""
    name, study = _split_path(path_image)
    remover = ObjectRemover(
        image_path=path_image,
        json_path=output_json,
        output_dir=OPERATION_DIR,
        expand_percent=30,
        output_path=os.path.join(OPERATION_DIR, f"{study}_{name}.png"),
    )
    out_path = remover.run()
    print("[remove] Готово")
    time.sleep(1)
    return out_path


@safe_step("detect_artifacts")
def step_detect_artifacts(report: StudyReport, path_image: str, clean_image: str) -> None:
    """Шаг 6. Поиск артефактов."""
    extractor = FragmentExtractor(clean_image)
    clf = FragmentClassifier(db_path=SPINE_PT, target=64, k=1)
    results = extractor.run()
    print("[fragments] Готово")

    bboxes = [
        bb['bbox_global_x1y1x2y2']
        for v in results.values()
        for bb in v['merged_bboxes']
    ]

    results, resultss = clf.classify_fragments(path_image, bboxes, return_details=True)
    print("artef", resultss)
    report.artifacts_present = resultss
    # if results["label"] == True:
    #     report.artifacts_present = True
    # else:
    #     report.artifacts_present = False

    path = Path(path_image)
    folder = path.parts[1]

    save_path = os.path.join(OUTPUT_DIR, f"{folder}_result.png")
    clf.visualize_fragments(
        path_image, bboxes,
        results=results,
        only_true=True,
        true_label='true',
        save_path=save_path,
    )


@safe_step("check_position")
def step_check_position(report: StudyReport, path_image, output_json: str) -> None:
    """Шаг 7. Проверка корректности положения."""
    taz = detect_pelvis(path_image)
    print(f"[pelvis] has_pelvis = {taz['has_pelvis']}")

    analyzer = SpineAnalyzer(path_image, output_json)
    result = analyzer.analyze()
    for b in result["vertebrae"]:
        print(f"  #{b['id']}: center=({b['cx']},{b['cy']}) "
              f"bbox=({b['x']},{b['y']},{b['w']}x{b['h']})")

    report.taz = taz["has_pelvis"]
    report.pozv = result["n_vertebrae"]

    if result["n_vertebrae"] > 5 and taz["has_pelvis"] is True:
        report.position_correct = True
    else:
        report.position_correct = False

    overlay_path = analyzer.save_overlay()
    print(f"[overlay] saved: {overlay_path}")



def analyze_spine(path_image: str) -> StudyReport:
    """Полный пайплайн с обработкой ошибок на каждом шаге."""
    report = StudyReport()
    t_start = time.perf_counter()
    print("=" * 60)
    print(f"[start] Анализ: {path_image}")

    # 1. Центрирование
    step_check_centering(report, path_image)

    # 2. Сегментация (нужна для остальных шагов)
    output_json = step_segmentation(report, path_image)
    if not output_json or not os.path.exists(output_json):
        print("[pipeline] Сегментация не дала результата — прерываем пайплайн")
        report.status = "failed"
        report.processing_time_sec = round(time.perf_counter() - t_start, 3)
        report.print_summary()
        return report

    # 3. Угол
    step_estimate_angle(report, path_image, output_json)

    # 4. Удаление позвоночника (нужно для поиска артефактов)
    clean_image = step_remove_spine(report, path_image, output_json)

    # 5. Артефакты (только если есть очищенное изображение)
    if clean_image and os.path.exists(clean_image):
        step_detect_artifacts(report, path_image, clean_image)
    else:
        report.add_error("detect_artifacts",
                         RuntimeError("нет очищенного изображения — шаг пропущен"))

    # 6. Положение
    step_check_position(report, path_image, output_json)

    # --- итоговый статус ---
    if not report.errors:
        report.status = "ok"
    elif any(v is not None for v in
             (report.center_offset, report.spine_angle, report.position_correct)):
        report.status = "partial"
    else:
        report.status = "failed"

    report.processing_time_sec = round(time.perf_counter() - t_start, 3)
    print(f"[done] Статус: {report.status}, время: {report.processing_time_sec} с")
    print("=" * 60)
    return report


def sustavec(img_path, report):
    criteria, side, rasst = pipeline_sustav.main(img_path)
    if criteria == 1:
        rota = "vnutr rotation"
    elif criteria == 2:
        rota = "naruzh rotation"
    else:
        rota = "norma"
    report.image_name = os.path.basename(img_path)
    path = Path(img_path)
    folder = path.parts[1]
    report.research_name = folder
    report.area = "sustav"
    report.sust_posit = rasst
    report.sust_side = side
    report.sust_rotation_std = rota

    return report


def main(path_image: str) -> Optional[StudyReport]:
    clear_dirs(dirs=None, recreate=True)
    """Точка входа: классификация → пайплайн → сохранение отчёта."""
    report = StudyReport()

    # Классификация
    label = step_classify(report, path_image)
    if label != "true":
        report = sustavec(path_image, report)
        # print(f"[main] Не позвоночник (label={label})")
        # report.status = "skipped"
        name, study = _split_path(path_image)
        path = Path(path_image)
        folder = path.parts[1]
        out_json = os.path.join(JSON_DIR_report, f"{folder}_{name}_report.json")
        report.save(out_json)
        print(f"[save] Отчёт сохранён: {out_json}")
        return report

    # Основной пайплайн
    report = analyze_spine(path_image)

    # Сохранение отчёта
    name, study = _split_path(path_image)
    path = Path(path_image)
    folder = path.parts[1]
    out_json = os.path.join(JSON_DIR_report, f"{folder}_{name}_report.json")
    report.save(out_json)
    print(f"[save] Отчёт сохранён: {out_json}")
    return report

if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(r"upload_dicom")
    DIR = ("upload_dicom")
    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".dcm"}


    for path in BASE_DIR.rglob("*"):
        try:
            remover = ImageDuplicateRemover(path)
            remover.remove_duplicates()
        except:
            pass


    for file_path in BASE_DIR.rglob("*"):  # rglob — рекурсивно
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXT:
            print(f"Обработка: {file_path}")
            main(str(file_path))


