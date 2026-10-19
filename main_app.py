import os
os.environ["TORCHINDUCTOR_TRITON_DISABLE_DEVICE_DETECTION"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # Дополнительная страховка
import torch
torch.set_num_threads(1)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

from find_copy import ImageDuplicateRemover


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models")
from datetime import datetime

warnings.filterwarnings(
    "ignore",
    message=r".*Overwriting tiny_vit_\d+m_\d+ in registry.*",
    category=UserWarning,
)
import os
import shutil
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, render_template_string
from flask import Flask, request, jsonify, make_response, send_file
import pipeline
import excel_import
import json
from pathlib import Path
from collections import defaultdict
import glob

import find_copy
import upload_help

app = Flask(__name__)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


BASE_DIR = Path(__file__).resolve().parent
UP_DIR   = BASE_DIR / "up_dicom"
UP_DIR_JPG   = BASE_DIR / "upload_dicom"
UP_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".dcm", ".dicom", ".zip", ""}  # "" — DICOM без расширения
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 МБ
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

JSON_REP_DIR = os.path.join(os.getcwd(), "json_rep")
ARTEFACTS_IMG_DIR = os.path.join(os.getcwd(), "static", "artefacts_img")


def _clear_dir_files(directory: str, extensions: tuple = None) -> int:
    """Удаляет файлы из папки. Если extensions задан — только с этими расширениями."""
    if not os.path.isdir(directory):
        return 0

    removed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if extensions is not None and not name.lower().endswith(extensions):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            app.logger.warning("Не удалось удалить %s: %s", path, e)
    return removed


def _clear_dir_all_except(directory: str, keep: tuple = ("__init__.py", "__init__.pyc")) -> int:
    """Удаляет всё содержимое папки (файлы и подпапки), кроме имён из keep."""
    if not os.path.isdir(directory):
        return 0

    keep_lower = {k.lower() for k in keep}
    removed = 0
    for name in os.listdir(directory):
        if name.lower() in keep_lower:
            continue
        path = os.path.join(directory, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError as e:
            app.logger.warning("Не удалось удалить %s: %s", path, e)
    return removed


@app.route("/reports/clear", methods=["POST"])
def clear_reports():
    try:
        json_removed = _clear_dir_files(JSON_REP_DIR, extensions=(".json",))
        img_removed = _clear_dir_files(
            ARTEFACTS_IMG_DIR,
            extensions=(".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"),
        )
        upload_removed = _clear_dir_all_except(
            UP_DIR_JPG,
            keep=("__init__.py", "__init__.pyc"),
        )

        return jsonify({
            "ok": True,
            "json_removed": json_removed,
            "images_removed": img_removed,
            "upload_removed": upload_removed,
        })
    except Exception as e:
        app.logger.exception("Ошибка очистки отчётов")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    upload_help.clear_up_dir_jpg()
    files = request.files.getlist("files")
    mode = request.form.get("mode", "files")

    if not files or all(not f.filename for f in files):
        return jsonify({"error": "Файлы не переданы"}), 400

    # Временная папка только под DICOM. После обработки будет удалена.
    work_dir = Path(tempfile.mkdtemp(prefix="req_", dir=str(UP_DIR)))

    # Папка для отдельных файлов (генерируется по дате и времени)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_1 = UP_DIR_JPG / f"upload_{timestamp}"
    out_dir_1.mkdir(parents=True, exist_ok=True)

    out_dir = UP_DIR_JPG
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = []

    try:
        for f in files:
            if not f.filename:
                continue

            rel = upload_help._sanitize_rel_path(f.filename)
            save_path = work_dir / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            f.save(str(save_path))

            if save_path.suffix.lower() == ".zip":
                unzip_dir = work_dir / f"unzip_{save_path.stem}"
                unzip_dir.mkdir(exist_ok=True)
                try:
                    upload_help.process_zip(save_path, unzip_dir, out_dir)
                    processed += 1
                except Exception as e:
                    app.logger.exception("zip failed")
                    errors.append(f"{f.filename}: {e}")

        if mode in ("folder"):
            # Обходим всё дерево DICOM и складываем JPG в out_dir/<UID>/<name>.jpg
            upload_help.process_folder(work_dir, out_dir)
            processed = sum(1 for _ in out_dir.rglob("*.jpg"))
        elif mode in ("files"):
            upload_help.process_folder(work_dir, out_dir_1)
            processed = sum(1 for _ in out_dir_1.rglob("*.jpg"))


    finally:
        # DICOM-дерево удаляем, JPG остаются в out_dir
        shutil.rmtree(work_dir, ignore_errors=True)

    if processed == 0 and not errors:
        return jsonify({"error": "Не удалось обработать файлы"}), 400

    for subdir in out_dir.iterdir():
        if subdir.is_dir():
            remover = find_copy.ImageDuplicateRemover(subdir)
            remover.remove_duplicates()
            print("Папка:", subdir)


    msg = f"Обработано файлов: {processed}"
    if errors:
        msg += f". Предупреждения ({len(errors)}): {'; '.join(errors[:3])}"
        if len(errors) > 3:
            msg += f" и ещё {len(errors) - 3}"

    status = 200 if processed > 0 else 400
    return jsonify({"message": msg, "processed": processed, "errors": errors}), status


@app.route("/start")
def media_counter_page():
    """Страница со счётчиком файлов и кнопкой запуска обработки."""
    total_files = upload_help.count_files_in_dir(UP_DIR_JPG)
    print("total",total_files)

    return render_template(
        "media_counter.html",
        total_files=total_files,
        up_dir=UP_DIR_JPG,
    )

def collect_reports(base_dir: str = "json_rep", limit_per_name: int = 3):
    """
    Возвращает список словарей с содержимым JSON-файлов.
    Для каждого 'имени' (префикса до _CR...) берём максимум limit_per_name файлов.
    """
    BASE_DIR = Path(base_dir)
    grouped = defaultdict(list)

    for file_path in BASE_DIR.rglob("*_report.json"):
        if not file_path.is_file():
            continue

        # имя = всё до последнего "_CR"
        # "2.25...450_CR000003_report.json" -> "2.25...450"
        stem = file_path.stem                      # убирает .json
        name = stem.split("_CR")[0]                # берём префикс

        grouped[name].append(file_path)

    result = []
    for name, files in grouped.items():
        # сортируем, чтобы порядок был стабильным (по суффиксу CRxxxxxx)
        files.sort(key=lambda p: p.stem)
        for fp in files[:limit_per_name]:          # ← максимум 3 на одно имя
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                data = {"error": str(e)}

            result.append({
                "name": name,
                "file": fp.name,
                "data": data,
            })

    return result

@app.route("/media-counter/start", methods=["POST"])
def media_counter_start():
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
            pipeline.main(str(file_path))
    return {"status": "success"}

@app.route("/reports")
def reports():
    return jsonify(collect_reports())

@app.route("/reports/view")
def reports_view():
    return render_template("reports.html", reports=collect_reports())

@app.route("/export", methods=["POST", "GET"])
def export():
    try:
        output_path = "report_1.xlsx"
        # 1. Собираем Excel из всех json-отчётов
        # covert_xls.json_to_xlsx(paths, output_path)
        paths = glob.glob("json_rep/*.json")
        df = excel_import.json_files_to_excel(paths, output_path)

        # import glob
        #
        # paths = glob.glob("json_rep/*.json")
        # df = json_files_to_excel(paths, "report1.xlsx")
        print(df)



        if not os.path.exists(output_path):
            return jsonify({"status": "error", "message": "Файл не создан"}), 500

        # 2. Отдаём файл на скачивание
        return send_file(
            output_path,
            as_attachment=True,
            download_name="report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500





@app.route("/examples")
def examples():
    folder = "examples"  # папка с JSON-файлами
    reports = [
        {
            "name": "2.25.66397484472966116002950503539482860450",
            "sustav": os.path.join(folder, "position.png")
        }    ]

    for report in reports:
        path = os.path.join(folder, "2.25.66397484472966116002950503539482860450_CR000000_report.json")
        with open(path, "r", encoding="utf-8") as f:
            report["data"] = json.load(f)
    return render_template("examples.html", reports=reports)

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Файл(ы) превышают допустимый размер"}), 413



@app.route('/artefacts')
def artefacts_gallery():
    if not os.path.exists(ARTEFACTS_IMG_DIR):
        return f"Папка {ARTEFACTS_IMG_DIR} не найдена.", 404

    # Достаем все файлы картинок
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    files = [
        f for f in os.listdir(ARTEFACTS_IMG_DIR)
        if f.lower().endswith(valid_extensions)
    ]
    files.sort()

    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Каталог артефактов</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 20px; }
            h1 { text-align: center; color: #333; }
            .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
            .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; transition: transform 0.2s; }
            .card:hover { transform: scale(1.02); }
            .card img { max-width: 100%; height: auto; border-radius: 4px; display: block; }
            .filename { margin-top: 10px; font-size: 12px; color: #666; word-break: break-all; }
        </style>
    </head>
    <body>
        <a href="/">На главную</a>

        <h1>Каталог файлов: artefacts_img</h1>
        <div class="gallery">
            {% for image in images %}
            <div class="card">
                <a href="{{ url_for('static', filename='artefacts_img/' + image) }}" target="_blank">
                    <img src="{{ url_for('static', filename='artefacts_img/' + image) }}" alt="{{ image }}">
                </a>
                <div class="filename">{{ image }}</div>
            </div>
            {% else %}
            <p>В папке нет файлов.</p>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, images=files)








if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)