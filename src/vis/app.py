import os
from glob import glob
from typing import List

import cv2
import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pydicom
import tifffile
from PIL import Image
from skimage import measure

from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS, CLASS_IDS_REVERSED


def get_img_show(
    img_num: int = 0,
    classes_vis: List[str] = None,
    opacity: int = 20,
):
    opacity *= 0.01
    opacity = 1 - opacity
    images = os.listdir('data/demo_2/input')
    img = Image.open(f'data/demo_2/input/{images[img_num]}')
    new_img = Image.new('RGB', (img.size[0] * 2, img.size[1]))
    color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
    new_img.paste(img, (0, 0))
    new_img.paste(color_mask, (img.size[0], 0))
    fig = px.imshow(new_img, height=img.size[1])
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    if classes_vis is not None and len(classes_vis) > 0:
        masks = tifffile.imread(f'data/demo_2/mask/{images[img_num].split(".")[0]}.tiff')
        for idx in CLASS_IDS_REVERSED:
            if CLASS_IDS_REVERSED[idx] in classes_vis:
                mask = masks[:, :, idx - 1].astype('uint8')
                area = np.nonzero(mask)
                if len(area) > 0:
                    area = pow(len(area[0]), 0.5)
                    area = area // 8
                    contours = measure.find_contours(mask, 0.5)
                    for contour in contours:
                        y, x = contour.T - 1
                        hover_info = (
                            '<br>' + f'{CLASS_IDS_REVERSED[idx]}/area: {area}' + ' <extra></extra>'
                        )
                        fig.add_scatter(
                            x=x,
                            y=y,
                            opacity=opacity,
                            mode='lines',
                            fill='toself',
                            showlegend=False,
                            hoveron='points+fills',
                            marker=dict(
                                color='#%02x%02x%02x' % CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idx]],
                            ),
                            hovertemplate=hover_info,
                            name=f'{CLASS_IDS_REVERSED[idx]}/area: {area}',
                        )
                        fig.add_scatter(
                            x=x + img.size[0],
                            y=y,
                            opacity=1.0,
                            mode='lines',
                            fill='toself',
                            showlegend=False,
                            hoveron='points+fills',
                            marker=dict(
                                color='#%02x%02x%02x' % CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idx]],
                            ),
                            hovertemplate=hover_info,
                            name=f'{CLASS_IDS_REVERSED[idx]}/area: {area}',
                        )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
    )
    return fig


def get_graph():
    masks = sorted(glob('data/demo_2/mask/*.tiff'))
    classes = {idx: [] for idx in CLASS_IDS_REVERSED}
    for idx, mask_path in enumerate(masks):
        mask = tifffile.imread(mask_path)
        for idy in CLASS_IDS_REVERSED:
            if np.unique(mask[:, :, idy - 1]).shape[0] == 2:
                classes[idy].append(idx)
    fig = go.Figure()
    for idy in CLASS_IDS_REVERSED:
        if len(classes[idy]) > 0:
            traces = []
            trace = [classes[idy][0]]
            for el in classes[idy][1:]:
                if el == trace[-1] + 1:
                    trace.append(el)
                else:
                    traces.append(trace)
                    trace = [el]
            traces.append(trace)
            for trace in traces:
                fig.add_trace(
                    go.Scatter(
                        x=trace,
                        y=[CLASS_IDS_REVERSED[idy] for _ in range(len(trace))],
                        marker=dict(
                            color='#%02x%02x%02x' % CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idy]],
                        ),
                    ),
                )
    fig.update_layout(showlegend=False)
    return fig


def get_graph_1(data):
    fig = go.Figure()
    fig_2 = go.Figure()
    for class_name in data:
        if len(data[class_name]['object_id']) > 0:
            object_id = data[class_name]['object_id'][0]
            trace = []
            for idx, object_id_ in enumerate(data[class_name]['object_id']):
                if object_id_ == object_id:
                    trace.append(
                        (data[class_name]['slice'][idx], data[class_name]['area'][idx]),
                    )
                else:
                    trace = np.array(trace)
                    if len(trace) >= 3:
                        fig.add_trace(
                            go.Scatter(
                                x=list(trace[:, 0]),
                                y=list(trace[:, 1]),
                                marker=dict(
                                    color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                                ),
                                name=f'{class_name}, {object_id}',
                            ),
                        )

                        fig_2.add_box(
                            y=list(trace[:, 1]),
                            name=f'{class_name}, {object_id}',
                            marker=dict(
                                color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                            ),
                        )

                    object_id = object_id_
                    trace = [(data[class_name]['slice'][idx], data[class_name]['area'][idx])]

            trace = np.array(trace)
            if len(trace) >= 3:
                fig.add_trace(
                    go.Scatter(
                        x=list(trace[:, 0]),
                        y=list(trace[:, 1]),
                        marker=dict(
                            color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                        ),
                        name=f'{class_name}, {object_id}',
                    ),
                )
                fig_2.add_box(
                    y=list(trace[:, 1]),
                    name=f'{class_name}, {object_id}',
                    marker=dict(
                        color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                    ),
                )
    fig.update_layout(showlegend=False)
    fig_2.update_layout(showlegend=False)
    return fig, fig_2


def get_analysis(
    file,
    progress=gr.Progress(),
):
    # TODO: inference model (dicom file analysis)
    study = pydicom.dcmread(file)
    dcm = study.pixel_array
    slices = dcm.shape[0]
    data = {
        class_name: {
            'area': [],
            'slice': [],
            'object_id': [],
        }
        for class_name in CLASS_IDS
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
                if len(data[CLASS_IDS_REVERSED[idy]]['object_id']) == 0:
                    data[CLASS_IDS_REVERSED[idy]]['object_id'].append(0)
                else:
                    if idx == data[CLASS_IDS_REVERSED[idy]]['slice'][-1] + 1:
                        data[CLASS_IDS_REVERSED[idy]]['object_id'].append(
                            data[CLASS_IDS_REVERSED[idy]]['object_id'][-1],
                        )
                    else:
                        data[CLASS_IDS_REVERSED[idy]]['object_id'].append(
                            data[CLASS_IDS_REVERSED[idy]]['object_id'][-1] + 1,
                        )
                data[CLASS_IDS_REVERSED[idy]]['slice'].append(idx)
                area = np.nonzero(mask[:, :, idy - 1])
                area = pow(len(area[0]), 0.5)
                area = area // 8
                data[CLASS_IDS_REVERSED[idy]]['area'].append(area)

    images = sorted(glob('data/demo_2/input/*.[pj][np][ge]*'))
    fig_1, fig_2 = get_graph_1(data)
    return (
        get_graph(),
        gr.Slider(minimum=0, maximum=len(images), value=0, visible=True, label='Номер кадра'),
        gr.Plot(
            visible=True,
            value=get_img_show(
                img_num=0,
                classes_vis=['Lumen', 'Lipid core', 'Fibrous cap', 'Vasa vasorum'],
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
            choices=(
                'Lumen',
                'Lipid core',
                'Fibrous cap',
                'Vasa vasorum',
            ),
            value=[
                'Lumen',
                'Lipid core',
                'Fibrous cap',
                'Vasa vasorum',
            ],
            visible=True,
        ),
        gr.Slider(
            value=20,
            minimum=0,
            maximum=100,
            label='Прозрачность, %',
            visible=True,
        ),
        fig_1,
        fig_2,
        gr.JSON(label='Metadata', value=data),
    )


def main():
    with gr.Blocks(title='KCC OCT analysis', theme=gr.themes.Origin(), fill_height=True) as block:
        gr.Markdown(
            """
          ## KCC: OCT analysis
      """,
        )
        with gr.Tab(label='UX test'):
            with gr.Row(variant='panel'):
                with gr.Column(scale=1):
                    with gr.Row():
                        input_data = gr.File(value='data/demo_2/IMG001', label='Исходный файл')
                    with gr.Row():
                        analysis = gr.Button('Провести анализ', variant='primary')
                with gr.Column(scale=3):
                    graph = gr.Plot()
            with gr.Row():
                with gr.Column(variant='panel'):
                    with gr.Row():
                        slider = gr.Slider(visible=False)
                    with gr.Row():
                        with gr.Column(scale=5):
                            img_show = gr.Plot(visible=False, container=False)
                        with gr.Column(scale=1):
                            with gr.Group():
                                params_mark = gr.Markdown(
                                    visible=False,
                                )
                                with gr.Row():
                                    classes = gr.Checkboxgroup(
                                        visible=False,
                                    )
                                with gr.Row():
                                    transparency = gr.Slider(
                                        visible=False,
                                    )
                    with gr.Row(variant='panel'):
                        areas_line = gr.Plot()
                    with gr.Row(variant='panel'):
                        areas_plot = gr.Plot()
                    with gr.Row(variant='panel'):
                        metadata = gr.JSON(label='Metadata')
            analysis.click(
                fn=get_analysis,
                inputs=input_data,
                outputs=[
                    graph,
                    slider,
                    img_show,
                    params_mark,
                    classes,
                    transparency,
                    areas_line,
                    areas_plot,
                    metadata,
                ],
            )
            slider.change(
                get_img_show,
                inputs=[
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
            )
            classes.change(
                get_img_show,
                inputs=[
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
            )
            transparency.change(
                get_img_show,
                inputs=[
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
            )
        with gr.Tab(label='Inference mode'):
            with gr.Row(variant='panel'):
                with gr.Column(scale=1):
                    with gr.Row():
                        input_data = gr.File()
                    with gr.Row():
                        analysis = gr.Button('Провести анализ', variant='primary')
                with gr.Column(scale=3):
                    graph = gr.Plot()
            with gr.Row():
                slider = gr.Slider(minimum=0, maximum=len(os.listdir('data/demo_2/input')))
            with gr.Row():
                img_show = gr.Image()
            # with gr.Row():

            with gr.Row():
                run = gr.Button()
            slider.change(
                get_img_show,
                inputs=slider,
                outputs=img_show,
            )
        run.click(
            get_graph,
            inputs=None,
            outputs=graph,
        )

    block.launch(
        server_name='0.0.0.0',
        server_port=7883,
        favicon_path='data/logo.ico',
        share=False,
    )


if __name__ == '__main__':
    main()
