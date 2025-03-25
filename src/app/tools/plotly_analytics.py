import numpy as np
import plotly.graph_objs as go

from src.data.utils import CLASS_COLORS_RGB


def get_object_map(data):
    # masks = sorted(glob('data/demo_2/mask/*.tiff'))
    # classes = {idx: [] for idx in CLASS_IDS_REVERSED}
    # for idx, mask_path in enumerate(masks):
    #     mask = tifffile.imread(mask_path)
    #     for idy in CLASS_IDS_REVERSED:
    #         if np.unique(mask[:, :, idy - 1]).shape[0] == 2:
    #             classes[idy].append(idx)
    fig = go.Figure()
    for class_name in data['objects']:
        traces = []
        if len(data['objects'][class_name]['object_id']) > 0:
            object_id = data['objects'][class_name]['object_id'][0]
            trace = []
            for idx, object_id_ in enumerate(data['objects'][class_name]['object_id']):
                if object_id_ == object_id:
                    trace.append(
                        data['objects'][class_name]['slice'][idx],
                    )
                else:
                    traces.append(trace)
                    trace = [data['objects'][class_name]['slice'][idx]]
            traces.append(trace)
            for trace in traces:
                fig.add_trace(
                    go.Scatter(
                        x=trace,
                        y=[class_name for _ in range(len(trace))],
                        marker=dict(
                            color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                        ),
                    ),
                )
    fig.update_layout(showlegend=False)
    return fig

    # for idy in CLASS_IDS_REVERSED:
    #     if len(classes[idy]) > 0:
    #         traces = []
    #         trace = [classes[idy][0]]
    #         for el in classes[idy][1:]:
    #             if el == trace[-1] + 1:
    #                 trace.append(el)
    #             else:
    #                 traces.append(trace)
    #                 trace = [el]
    #         traces.append(trace)
    #         for trace in traces:
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=trace,
    #                     y=[CLASS_IDS_REVERSED[idy] for _ in range(len(trace))],
    #                     marker=dict(
    #                         color='#%02x%02x%02x' % CLASS_COLORS_RGB[CLASS_IDS_REVERSED[idy]],
    #                     ),
    #                 ),
    #             )
    # fig.update_layout(showlegend=False)
    # return fig


def get_trace_area(classes, data):
    fig = go.Figure()
    for class_name in data['objects']:
        if class_name in classes:
            if len(data['objects'][class_name]['object_id']) > 0:
                object_id = data['objects'][class_name]['object_id'][0]
                object_idx = 1
                trace = []
                for idx, object_id_ in enumerate(data['objects'][class_name]['object_id']):
                    if object_id_ == object_id:
                        trace.append(
                            (
                                data['objects'][class_name]['slice'][idx],
                                data['objects'][class_name]['area'][idx],
                            ),
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
                                    name=f'{class_name}, {object_idx}',
                                ),
                            )
                            object_idx += 1

                        object_id = object_id_
                        trace = [
                            (
                                data['objects'][class_name]['slice'][idx],
                                data['objects'][class_name]['area'][idx],
                            ),
                        ]
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
    fig.update_layout(
        showlegend=False,
        xaxis_title='Samples',
        yaxis_title='Area',
    )
    return fig


def get_plot_area(classes, data):
    fig = go.Figure()
    for class_name in data['objects']:
        if class_name in classes:
            if len(data['objects'][class_name]['object_id']) > 0:
                object_id = data['objects'][class_name]['object_id'][0]
                object_idx = 1
                trace = []
                for idx, object_id_ in enumerate(data['objects'][class_name]['object_id']):
                    if object_id_ == object_id:
                        trace.append(
                            (
                                data['objects'][class_name]['slice'][idx],
                                data['objects'][class_name]['area'][idx],
                            ),
                        )
                    else:
                        trace = np.array(trace)
                        if len(trace) >= 3:
                            fig.add_box(
                                y=list(trace[:, 1]),
                                name=f'{class_name}, {object_idx}',
                                marker=dict(
                                    color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                                ),
                            )
                            object_idx += 1

                        object_id = object_id_
                        trace = [
                            (
                                data['objects'][class_name]['slice'][idx],
                                data['objects'][class_name]['area'][idx],
                            ),
                        ]
                trace = np.array(trace)
                if len(trace) >= 3:
                    fig.add_box(
                        y=list(trace[:, 1]),
                        name=f'{class_name}, {object_idx}',
                        marker=dict(
                            color='#%02x%02x%02x' % CLASS_COLORS_RGB[class_name],
                        ),
                    )
    fig.update_layout(
        showlegend=False,
        xaxis_title='Objects',
        yaxis_title='Area',
    )
    return fig
