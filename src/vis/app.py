import os

import gradio as gr

from src.data.utils import CLASS_IDS
from src.vis.tools.analysis import get_analysis
from src.vis.tools.img_viewer import get_img_show
from src.vis.tools.plotly_analytics import get_plot_area, get_trace_area


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
                        with gr.Column(scale=5):
                            areas_line = gr.Plot()
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown(
                                    """
                                      # Параметры
                                    """,
                                )
                                with gr.Row():
                                    classes_trace = gr.Checkboxgroup(
                                        label='Объекты',
                                        choices=[class_name for class_name in CLASS_IDS],
                                        value=[class_name for class_name in CLASS_IDS],
                                    )
                    with gr.Row(variant='panel'):
                        with gr.Column(scale=5):
                            areas_plot = gr.Plot()
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown(
                                    """
                                      # Параметры
                                    """,
                                )
                                with gr.Row():
                                    classes_plot = gr.Checkboxgroup(
                                        label='Объекты',
                                        choices=[class_name for class_name in CLASS_IDS],
                                        value=[class_name for class_name in CLASS_IDS],
                                    )
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
            classes_trace.change(
                get_trace_area,
                inputs=[
                    classes_trace,
                    metadata,
                ],
                outputs=areas_line,
            )
            classes_plot.change(
                get_plot_area,
                inputs=[
                    classes_plot,
                    metadata,
                ],
                outputs=areas_plot,
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
        # run.click(
        #     get_analysis,
        #     inputs=None,
        #     outputs=graph,
        # )

    block.launch(
        server_name='0.0.0.0',
        server_port=7883,
        favicon_path='data/logo.ico',
        share=False,
    )


if __name__ == '__main__':
    main()
