import json
import numpy as np
import matplotlib.pyplot as plt


def analyze_hip_rotation(json_path):
    # 1. Загрузка данных
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Очистка ключей от случайных пробелов в конце (особенность вашего JSON)
    data = {k.strip(): v for k, v in data.items()}
    shape = data['shapes'][0]
    shape = {k.strip(): v for k, v in shape.items()}

    points = np.array(shape['points'])
    x = points[:, 0]
    y = points[:, 1]

    # 2. Определение зон интереса
    y_min, y_max = y.min(), y.max()
    height = y_max - y_min

    # Зона диафиза (ствола) бедра - нижние 20% полигона
    shaft_mask = y > (y_max - 0.20 * height)
    shaft_x = x[shaft_mask]
    shaft_y = y[shaft_mask]

    # ИЗМЕНЕНИЕ: Нога справа, значит медиальный край - это ПРАВЫЙ (максимальный X)
    shaft_medial_x = shaft_x.max()
    shaft_lateral_x = shaft_x.min()
    shaft_width = shaft_medial_x - shaft_lateral_x

    # 3. Поиск малого вертела
    # Зона от 20% до 50% от низа снимка
    lt_mask = (y > (y_max - 0.50 * height)) & (y <= (y_max - 0.20 * height))

    if lt_mask.sum() > 0:
        lt_x = x[lt_mask]
        lt_y = y[lt_mask]

        # ИЗМЕНЕНИЕ: Ищем самую медиальную точку (максимальный X)
        lesser_trochanter_max_x = lt_x.max()
        lt_point_idx = np.argmax(lt_x)
        lt_point = (lt_x[lt_point_idx], lt_y[lt_point_idx])
    else:
        lesser_trochanter_max_x = shaft_medial_x
        lt_point = (shaft_medial_x, y_max - 0.3 * height)

    # 4. Расчет Индекса Ротации
    # Насколько малый вертел выступает вправо за пределы медиального края диафиза
    protrusion = lesser_trochanter_max_x - shaft_medial_x
    rotation_index = protrusion / shaft_width if shaft_width > 0 else 0

    # # 5. Визуализация
    # fig, ax = plt.subplots(figsize=(8, 10))
    # ax.plot(x, y, 'b-', linewidth=2, label='Контур сустава (sustav)')
    # ax.fill(x, y, alpha=0.2, color='blue')
    #
    # # Отмечаем диафиз
    # ax.scatter(shaft_x, shaft_y, color='gray', s=10, label='Диафиз (ствол)')
    # ax.axvline(x=shaft_medial_x, color='green', linestyle='--', label=f'Медиальный край (x={shaft_medial_x:.1f})')
    #
    # # Отмечаем малый вертел
    # ax.scatter(*lt_point, color='red', s=100, zorder=5, label=f'Малый вертел (x={lesser_trochanter_max_x:.1f})')
    # ax.plot([shaft_medial_x, lesser_trochanter_max_x], [lt_point[1], lt_point[1]], 'r-', linewidth=2)
    #
    # ax.invert_yaxis()  # Инвертируем Y, так как на изображениях 0 сверху
    # ax.set_aspect('equal')
    # ax.set_title(f'Анализ ротации (нога справа)\nИндекс ротации: {rotation_index:.2f}')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    # 6. Интерпретация результата
    print(f"Ширина диафиза: {shaft_width:.1f} px")
    print(f"Выступание малого вертела: {protrusion:.1f} px")
    print(f"Индекс ротации (Выступание / Ширина): {rotation_index:.2f}\n")

    if rotation_index < 0.05:
        res = 1
        print("Оценка: Внутренняя ротация (малый вертел практически не виден).")
    elif rotation_index < 0.15:
        res = 0
        print("Оценка: Нейтральное положение (малый вертел слегка выступает).")
    else:
        res = 2
        print("Оценка: Наружная ротация (малый вертел четко и сильно выступает).")

    return res


# Запуск функции
if __name__ == "__main__":
    analyze_hip_rotation('series_002_2_CR_CR000000.json')