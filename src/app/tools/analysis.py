import base64
import math
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, cast

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.app.tools.img_viewer import get_img_show
from src.app.tools.plotly_analytics import get_plot_area, get_trace_area
from src.data.utils import CLASS_IDS, CLASS_IDS_REVERSED


def calculate_thickness_contour(
    mask: np.ndarray,
) -> Dict[str, Any]:
    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "median": 0,
            "min": 0,
            "max": 0,
            "all_measurements": [],
        }

    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)

    # Находим центр масс
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return {
            "median": 0,
            "min": 0,
            "max": 0,
            "all_measurements": [],
        }
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Рассчитываем расстояния от центра до всех точек контура
    distances = [np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2) for point in contour]

    return {
        "median": np.median(distances),
        "min": np.min(distances),
        "max": np.max(distances),
        "all_measurements": distances,
    }


def calculate_object_thickness(mask: np.ndarray) -> Dict[str, Any]:
    """Рассчитывает толщину объекта на бинарном изображении.

    Параметры:
    mask (numpy.ndarray): Изображение-маска (0 - фон, 255 - объект)

    Возвращает:
    dict: Словарь с медианной, минимальной, максимальной толщиной и всеми измерениями
    """
    # Проверка, что изображение одноканальное
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Находим центр изображения
    height, width = mask.shape
    center_x, center_y = width // 2, height // 2

    # Создаем массив для хранения длин радиусов
    radii = []

    # Перебираем углы от 0 до 360 градусов с шагом 1 градус
    for angle in range(0, 360, 1):
        # Преобразуем угол в радианы
        rad = math.radians(angle)

        # Инициализируем переменные для текущего радиуса
        current_radius = 0
        found_object = False

        # Проверяем пиксели вдоль радиуса (максимальная длина - диагональ изображения)
        max_radius = int(math.sqrt(width**2 + height**2)) // 2

        for r in range(1, max_radius):
            # Вычисляем координаты точки на радиусе
            x = int(center_x + r * math.cos(rad))
            y = int(center_y + r * math.sin(rad))

            # Проверяем, находится ли точка в пределах изображения
            if 0 <= x < width and 0 <= y < height:
                # Если пиксель принадлежит объекту
                if mask[y, x] == 255:
                    current_radius = r
                    found_object = True
                # Если вышли из объекта (для случая, когда объект не выпуклый)
                elif found_object:
                    break
            else:
                break

        if found_object:
            radii.append(current_radius)

    if not radii:
        return {
            "median": 0,
            "min": 0,
            "max": 0,
            "all_measurements": [],
        }

    # Рассчитываем статистику
    median_thickness = np.median(radii)
    min_thickness = np.min(radii)
    max_thickness = np.max(radii)

    return {
        "median": median_thickness,
        "min": min_thickness,
        "max": max_thickness,
        "all_measurements": radii,
    }


def get_info_panel(name: str, value: str, description: str):
    return (
        f""
        f'<div style="'
        f"border: 1px solid #ddd;"
        f"border-radius: 12px;"
        f"padding: 20px;"
        f"width: 100%;"
        f"height: 100%;"
        f"text-align: center;"
        f"box-shadow: 0 2px 5px rgba(0,0,0,0.05);"
        f'">'
        f'<div style="font-size: 22px; font-weight: 600; color: #E41EC7;">{name}</div>'
        f'<div style="font-size: 36px; font-weight: 700; color: #2E86DE; margin: 8px 0;">{value}</div>'
        f'<div style="font-size: 16px; color: #666;">{description}</div>'
        f"</div>"
        f""
    )


def get_analysis(
    file,
    inference_type: str,
    progress=gr.Progress(),
):
    study = np.load(file)["data"]
    slices = study.shape[0]

    # Typed storage for analysis results
    objects: Dict[str, Dict[str, List[Any]]] = {
        class_name: {
            "area": [],
            "thickness_mean": [],
            "thickness_min": [],
            "slice": [],
            "object_id": [],
            "masks": [],
        }
        for class_name in CLASS_IDS
    }
    ratio: int = int(study.shape[1] * 150 // 1000)

    data: Dict[str, Any] = {
        "ratio": ratio,
        "objects": objects,
    }

    class_ids_reversed_typed = cast(Dict[int, str], CLASS_IDS_REVERSED)
    images = []

    for idx, slice in enumerate(progress.tqdm(study, desc="Processing")):
        img = slice[:, :, :3]
        images.append(Image.fromarray(np.array(img).astype("uint8")))
        mask: np.ndarray = slice[:, :, 3:]
        for idy in class_ids_reversed_typed:
            class_name = class_ids_reversed_typed[idy]
            if np.unique(mask[:, :, idy - 1]).shape[0] == 2:
                obj = objects[class_name]
                if len(obj["object_id"]) == 0:
                    obj["object_id"].append(0)
                else:
                    if idx == obj["slice"][-1] + 1:
                        obj["object_id"].append(obj["object_id"][-1])
                    else:
                        obj["object_id"].append(obj["object_id"][-1] + 1)
                obj["slice"].append(idx)
                area_idx = np.nonzero(mask[:, :, idy - 1])
                area = pow(len(area_idx[0]) // ratio, 0.5)
                obj["area"].append(area)
                obj["thickness_mean"].append(
                    calculate_thickness_contour(mask[:, :, idy - 1])["median"] / ratio,
                )
                obj["thickness_min"].append(
                    calculate_thickness_contour(mask[:, :, idy - 1])["min"] / ratio,
                )
                buff = BytesIO()
                Image.fromarray(mask[:, :, idy - 1]).save(buff, format="png")
                im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
                obj["masks"].append(im_b64)
    fc_count = Counter(data["objects"]["Fibrous cap"]["object_id"])
    fc_unique_obj = [num for num, c in fc_count.items() if c >= 3]
    return (
        gr.Slider(minimum=0, maximum=slices, value=0, visible=True, label="Номер кадра"),
        gr.Plot(
            visible=True,
            value=get_img_show(
                img_num=0,
                classes_vis=[class_name for class_name in CLASS_IDS],
                opacity=20,
                data=data,
                images=images,
            ),
        ),
        gr.Markdown(
            """
              # Параметры
            """,
            visible=True,
        ),
        gr.Checkboxgroup(
            label="Объекты",
            choices=[class_name for class_name in CLASS_IDS],
            value=[class_name for class_name in CLASS_IDS],
            visible=True,
        ),
        gr.Slider(
            value=20,
            minimum=0,
            maximum=100,
            label="Прозрачность, %",
            visible=True,
        ),
        get_trace_area(classes=[class_name for class_name in CLASS_IDS], data=data),
        get_plot_area(classes=[class_name for class_name in CLASS_IDS], data=data),
        data,
        images,
        get_info_panel(
            "Lumen",
            f"{int(np.mean(data['objects']['Lumen']['area']))} ± {int(np.std((data['objects']['Lumen']['area'])))} мкм",
            "Площадь",
        ),
        get_info_panel(
            "Fibrous cap",
            f"{round(np.min(data['objects']['Fibrous cap']['thickness_mean']), 2)} мкм"
            if len(data["objects"]["Fibrous cap"]["thickness_mean"]) > 0
            else "-",
            "Минимальная толщина",
        ),
        get_info_panel(
            "Fibrous cap",
            f"{round(np.mean(data['objects']['Fibrous cap']['thickness_mean']), 2)} ± {round(np.std(data['objects']['Fibrous cap']['thickness_mean']), 2)} мкм"
            if len(data["objects"]["Fibrous cap"]["thickness_mean"]) > 0
            else "-",
            "Толщина",
        ),
        get_info_panel("Fibrous cap", f"{len(np.unique(fc_unique_obj))}", "Количество объектов"),
    )
