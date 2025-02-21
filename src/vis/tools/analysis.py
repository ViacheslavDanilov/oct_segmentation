from glob import glob

import cv2
import gradio as gr
import numpy as np
import pydicom
import tifffile

from src.data.utils import CLASS_IDS, CLASS_IDS_REVERSED
from src.vis.tools.img_viewer import get_img_show
from src.vis.tools.plotly_analytics import get_object_map, get_plot_area, get_trace_area


def get_analysis(
    file,
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
                'slice': [],
                'object_id': [],
            }
            for class_name in CLASS_IDS
        },
    }
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

    masks = sorted(glob('data/demo_2/mask/*.tiff'))
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

    images = sorted(glob('data/demo_2/input/*.[pj][np][ge]*'))
    return (
        get_object_map(data),
        gr.Slider(minimum=0, maximum=len(images), value=0, visible=True, label='Номер кадра'),
        gr.Plot(
            visible=True,
            value=get_img_show(
                img_num=0,
                classes_vis=[class_name for class_name in CLASS_IDS],
                opacity=20,
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
    )
