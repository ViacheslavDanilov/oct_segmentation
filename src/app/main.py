import os
from glob import glob

import gradio as gr

from src.app.tools.analysis import get_analysis, get_gpt_analysis
from src.app.tools.img_viewer import get_img_show
from src.app.tools.plotly_analytics import get_plot_area, get_trace_area
from src.data.utils import CLASS_IDS


def get_patient_files():
    """Get list of available patient files."""
    patient_dir = "data/app/demo/patients"
    files = sorted(glob(os.path.join(patient_dir, "*.npz")))
    return files


def main():
    with gr.Blocks(title="ОКТ Анализ", theme=gr.themes.Origin(), fill_height=True) as block:
        gr.Markdown(
            """
            ## Анализ оптической когерентной томографии
            """,
        )
        with gr.Tab(label="UX test"):
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    with gr.Row():
                        input_data = gr.Dropdown(
                            choices=get_patient_files(),
                            value="data/app/demo/patients/001.npz",
                            label="Исходные данные",
                            interactive=True,
                        )
                    with gr.Row():
                        analysis = gr.Button("Анализ", variant="primary")
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column(scale=1):
                            mean_area_lumen = gr.HTML()
                        # with gr.Column(scale=1):
                        #     min_size_FC = gr.HTML()
                        with gr.Column(scale=1):
                            mean_size_FC = gr.HTML()
                        with gr.Column(scale=1):
                            counter_FC = gr.HTML()
                        with gr.Column(scale=1):
                            risc_classification = gr.HTML()
            min_size_FC = gr.HTML(visible=False)
            with gr.Row():
                result_analysis_text = gr.Text(label="Результат анализа", lines=12)
            with gr.Row():
                with gr.Column(variant="panel"):
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
                    with gr.Row(variant="panel"):
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
                                        label="Объекты",
                                        choices=[class_name for class_name in CLASS_IDS],
                                        value=[class_name for class_name in CLASS_IDS],
                                    )
                    with gr.Row(variant="panel"):
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
                                        label="Объекты",
                                        choices=[class_name for class_name in CLASS_IDS],
                                        value=[class_name for class_name in CLASS_IDS],
                                    )
                    with gr.Row(variant="panel"):
                        metadata = gr.JSON(label="Metadata", visible=False)
                        images_ = gr.Gallery(visible=False, type="pil")
            analysis.click(
                fn=get_analysis,
                inputs=[input_data, gr.State("demo")],
                outputs=[
                    slider,
                    img_show,
                    params_mark,
                    classes,
                    transparency,
                    areas_line,
                    areas_plot,
                    metadata,
                    images_,
                    mean_area_lumen,
                    min_size_FC,
                    mean_size_FC,
                    counter_FC,
                ],
            )
            slider.change(
                get_img_show,
                inputs=[
                    metadata,
                    images_,
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
                show_progress="hidden",
            )
            classes.change(
                get_img_show,
                inputs=[
                    metadata,
                    images_,
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
                show_progress="hidden",
            )
            transparency.change(
                get_img_show,
                inputs=[
                    metadata,
                    images_,
                    slider,
                    classes,
                    transparency,
                ],
                outputs=img_show,
                show_progress="hidden",
            )
            classes_trace.change(
                get_trace_area,
                inputs=[
                    classes_trace,
                    metadata,
                ],
                outputs=areas_line,
                show_progress="hidden",
            )
            classes_plot.change(
                get_plot_area,
                inputs=[
                    classes_plot,
                    metadata,
                ],
                outputs=areas_plot,
                show_progress="hidden",
            )
            metadata.change(
                fn=get_gpt_analysis,
                inputs=[metadata],
                outputs=[result_analysis_text, risc_classification],
            )
    block.launch(
        server_name="0.0.0.0",
        server_port=7883,
        favicon_path="data/app/logo.ico",
        share=False,
    )


if __name__ == "__main__":
    main()
