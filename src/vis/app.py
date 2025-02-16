import os
from glob import glob

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import tifffile
from PIL import Image
from skimage import measure

from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS_REVERSED


def get_img_show(
    img_num,
):
    images = os.listdir('data/demo_2/input')
    img = Image.open(f'data/demo_2/input/{images[img_num]}')
    masks = tifffile.imread(f'data/demo_2/mask/{images[img_num].split(".")[0]}.tiff')
    color_mask = Image.new('RGB', size=(1024, 1024), color=(128, 128, 128))
    for idx in CLASS_IDS_REVERSED:
        class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idx]])
        mask = Image.fromarray(masks[:, :, idx - 1].astype('uint8'))
        color_mask.paste(class_img, (0, 0), mask)
    new_img = Image.new('RGB', (img.size[0] * 2, img.size[1]))
    new_img.paste(img, (0, 0))
    new_img.paste(color_mask, (img.size[0], 0))
    fig = px.imshow(new_img, height=1024)
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    # fig = go.Figure()
    # fig.add_trace(new_img)
    for idx in CLASS_IDS_REVERSED:
        # class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idx]])
        mask = Image.fromarray(masks[:, :, idx - 1].astype('uint8'))
        contours = measure.find_contours(np.array(mask), 0.5)
        for contour in contours:
            y, x = contour.T - 1
            fig.add_scatter(
                x=x,
                y=y,
                opacity=0.8,
                mode='lines',
                fill='toself',
                showlegend=False,
                hoveron='points+fills',
            )
        # mask = mask / 255.0
        # fig.add_trace(
        #     go.Contour(
        #         z=mask,
        #         showscale=True,
        #         # autocontour=True,
        #         contours=dict(coloring='lines'),
        #         # opacity=0.6,
        #         # line_width=1,
        #     ),
        # )
        # fig.add_trace(
        #     go.Image(
        #         z=mask,
        #         opacity=0.5
        #     )
        # )
        # fig.add_trace(
        #     go.Heatmap(z=mask, showscale=False, zmin=0, zmax=1)
        # )
    fig.update_layout(showlegend=False)
    return fig


def get_graph():
    # TODO: for CLASS_IDX, CLASS_COLOR, optimize, class plot
    masks = sorted(glob('data/demo_2/mask/*.tiff'))
    classes = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    for idx, mask_path in enumerate(masks):
        mask = tifffile.imread(mask_path)
        for i in range(4):
            if np.unique(mask[:, :, i]).shape[0] == 2:
                classes[i].append(idx)
    fig = go.Figure()
    for i in range(4):
        if len(classes[i]) > 0:
            traces = []
            trace = [classes[i][0]]
            for el in classes[i][1:]:
                if el == trace[-1] + 1:
                    trace.append(el)
                else:
                    traces.append(trace)
                    trace = [el]
            traces.append(trace)
            for trace in traces:
                if i == 0:
                    fig.add_trace(
                        go.Scatter(
                            x=trace,
                            y=['Lumen' for _ in range(len(trace))],
                            marker=dict(color='#636EFA'),
                        ),
                    )
                if i == 1:
                    fig.add_trace(
                        go.Scatter(
                            x=trace,
                            y=['Lipid core' for _ in range(len(trace))],
                            marker=dict(color='#19D3F3'),
                        ),
                    )
                if i == 2:
                    fig.add_trace(
                        go.Scatter(
                            x=trace,
                            y=['Fibrous cap' for _ in range(len(trace))],
                            marker=dict(color='#EF553B'),
                        ),
                    )
                if i == 3:
                    fig.add_trace(
                        go.Scatter(
                            x=trace,
                            y=['Vasa vasorum' for _ in range(len(trace))],
                            marker=dict(color='#FFA15A'),
                        ),
                    )
    fig.update_layout(showlegend=False)
    return fig


def get_analysis(
    file,
):
    # TODO: inference model (dicom file analysis)

    images = sorted(glob('data/demo_2/input/*.[pj][np][ge]*'))
    return (
        get_graph(),
        gr.Slider(minimum=0, maximum=len(images), value=0, visible=True, label='Номер кадра'),
        gr.Plot(visible=True, value=get_img_show(0)),
    )


def main():
    with gr.Blocks(title='KCC OCT analysis') as block:
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
                            with gr.Group(visible=True):
                                gr.Markdown(
                                    """
                                    # Параметры
                                    """,
                                )
                                with gr.Row():
                                    classes = gr.Checkboxgroup(
                                        label='Объекты',
                                        choices=(
                                            'Lumen',
                                            'Lipid core',
                                            'Fibrous cap',
                                            'Vasa vasorum',
                                        ),
                                    )
                                with gr.Row():
                                    transparency = gr.Slider(
                                        value=15,
                                        minimum=0,
                                        maximum=100,
                                        label='Прозрачность, %',
                                    )
            # with gr.Row():

            # with gr.Row():
            #     run = gr.Button()
            analysis.click(
                fn=get_analysis,
                inputs=input_data,
                outputs=[graph, slider, img_show],
            )
            slider.change(
                get_img_show,
                inputs=slider,
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
        # debug=True,
        server_name='0.0.0.0',
        server_port=7883,
        favicon_path='data/logo.ico',
        share=False,
    )


if __name__ == '__main__':
    main()
