import base64
import os
import uuid
from glob import glob
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
import pydicom
import tifffile
from PIL import Image

from src.app.tools.img_viewer import get_img_show
from src.app.tools.plotly_analytics import get_object_map, get_plot_area, get_trace_area
from src.data.utils import CLASS_IDS, CLASS_IDS_REVERSED


def calculate_thickness_contour(
    mask: np.array,
):
    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)

    # Находим центр масс
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Рассчитываем расстояния от центра до всех точек контура
    distances = [np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2) for point in contour]

    return {
        'median': np.median(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'all_measurements': distances
    }


def calculate_object_thickness(mask):
    """
    Рассчитывает толщину объекта на бинарном изображении.

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
        max_radius = int(math.sqrt(width ** 2 + height ** 2)) // 2

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
            'median': 0,
            'min': 0,
            'max': 0,
            'all_measurements': []
        }

    # Рассчитываем статистику
    median_thickness = np.median(radii)
    min_thickness = np.min(radii)
    max_thickness = np.max(radii)

    return {
        'median': median_thickness,
        'min': min_thickness,
        'max': max_thickness,
        'all_measurements': radii
    }


def get_analysis(
    file,
    inference_type: str,
    progress=gr.Progress(),
):
    # TODO: inference model (dicom file analysis)
    study = pydicom.dcmread(file)
    dcm = study.pixel_array
    slices = dcm.shape[0]
    data = {
        'ratio': int(dcm.shape[1] * 150 // 1000),
        'objects': {
            class_name: {
                'area': [],
                'thickness_mean': [],
                'thickness_min': [],
                'slice': [],
                'object_id': [],
                'masks': [],
                'img_name': [],
            }
            for class_name in CLASS_IDS
        },
        'images': []
    }
    if inference_type == 'demo':
        work_dir = 'data/app/demo'
        masks = sorted(glob(f'{work_dir}/mask/*.tiff'))
    else:
        work_dir = f'data/app/temp/{uuid.uuid4()}'
        for slice in progress.tqdm(range(slices), desc='Processing'):
            img = dcm[slice]
            img = cv2.normalize(
                img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, mask_path in enumerate(masks):
        mask = tifffile.imread(mask_path)
        for idy in CLASS_IDS_REVERSED:
            if np.unique(mask[:, :, idy - 1]).shape[0] == 2:
                if len(data['objects'][CLASS_IDS_REVERSED[idy]]['object_id']) == 0:
                    data['objects'][CLASS_IDS_REVERSED[idy]]['object_id'].append(0)
                else:
                    if idx == data['objects'][CLASS_IDS_REVERSED[idy]]['slice'][-1] + 1:
                        data['objects'][CLASS_IDS_REVERSED[idy]]['object_id'].append(
                            data['objects'][CLASS_IDS_REVERSED[idy]]['object_id'][-1],
                        )
                    else:
                        data['objects'][CLASS_IDS_REVERSED[idy]]['object_id'].append(
                            data['objects'][CLASS_IDS_REVERSED[idy]]['object_id'][-1] + 1,
                        )
                data['objects'][CLASS_IDS_REVERSED[idy]]['slice'].append(idx)
                area = np.nonzero(mask[:, :, idy - 1])
                area = pow(len(area[0]) // data['ratio'], 0.5)
                data['objects'][CLASS_IDS_REVERSED[idy]]['area'].append(area)
                data['objects'][CLASS_IDS_REVERSED[idy]]['thickness_mean'].append(
                    calculate_thickness_contour(mask[:, :, idy - 1])['median'] / data['ratio']
                )
                data['objects'][CLASS_IDS_REVERSED[idy]]['thickness_min'].append(
                    calculate_thickness_contour(mask[:, :, idy - 1])['min'] / data['ratio']
                )
                buff = BytesIO()
                Image.fromarray(mask[:, :, idy - 1]).save(buff, format='png')
                im_b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
                data['objects'][CLASS_IDS_REVERSED[idy]]['masks'].append(im_b64)
                data['objects'][CLASS_IDS_REVERSED[idy]]['img_name'].append(
                    os.path.basename(mask_path).split('.')[0]
                )
        data['images'].append(os.path.basename(mask_path).split('.')[0])
    return (
        get_object_map(data),
        gr.Slider(minimum=0, maximum=len(masks), value=0, visible=True, label='Номер кадра'),
        gr.Plot(
            visible=True,
            value=get_img_show(
                img_num=0,
                classes_vis=[class_name for class_name in CLASS_IDS],
                img_dir=f'{work_dir}/img',
                opacity=20,
                data=data,
            ),
        ),
        gr.Markdown(
            """
              # Параметры
            """,
            visible=True,
        ),
        gr.Checkboxgroup(
            label='Объекты',
            choices=[class_name for class_name in CLASS_IDS],
            value=[class_name for class_name in CLASS_IDS],
            visible=True,
        ),
        gr.Slider(
            value=20,
            minimum=0,
            maximum=100,
            label='Прозрачность, %',
            visible=True,
        ),
        get_trace_area(classes=[class_name for class_name in CLASS_IDS], data=data),
        get_plot_area(classes=[class_name for class_name in CLASS_IDS], data=data),
        gr.JSON(label='Metadata', value=data),
        f'{work_dir}/img'
    )
