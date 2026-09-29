import json
import glob
import pandas as pd


def json_to_xlsx(json_files_pattern: str, output_path: str = "output.xlsx") -> None:
    """
    Выгружает JSON-файлы в XLSX, исключая поле steps_time_sec.

    :param json_files_pattern: путь/маска для JSON-файлов, например "*.json" или "data/*.json"
    :param output_path: имя выходного XLSX-файла
    """
    rows = []
    for file_path in glob.glob(json_files_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # убираем ненужное поле
        data.pop("steps_time_sec", None)
        rows.append(data)

    if not rows:
        print("Не найдено ни одного JSON-файла по маске:", json_files_pattern)
        return

    df = pd.DataFrame(rows)

    # errors — список, преобразуем в строку для читаемости в Excel
    if "errors" in df.columns:
        df["errors"] = df["errors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    df.to_excel(output_path, index=False)
    print(f"Сохранено {len(rows)} записей в {output_path}")


if __name__ == "__main__":
    json_to_xlsx("json_rep/*.json", "report.xlsx")