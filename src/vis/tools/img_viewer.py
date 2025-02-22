import base64
from io import BytesIO
from typing import List

import numpy as np
import plotly.express as px
from PIL import Image
from skimage import measure

from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS_REVERSED


def get_img_show(
    data,
    img_num: int = 0,
    classes_vis: List[str] = None,
    opacity: int = 20,
):
    opacity *= 0.01
    opacity = 1 - opacity
    img = Image.open(
        f"data/demo_2/input/{data['objects'][classes_vis[0]]['img_name'][img_num]}.png"
    )
    new_img = Image.new('RGB', (img.size[0] * 2, img.size[1]))
    color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
    new_img.paste(img, (0, 0))
    new_img.paste(color_mask, (img.size[0], 0))
    fig = px.imshow(new_img, height=img.size[1])
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    if classes_vis is not None and len(classes_vis) > 0:
        for idx in CLASS_IDS_REVERSED:
            if CLASS_IDS_REVERSED[idx] in classes_vis:
                if img_num in data['objects'][CLASS_IDS_REVERSED[idx]]['slice']:
                    id = data['objects'][CLASS_IDS_REVERSED[idx]]['slice'].index(img_num)
                    mask_b64 = data['objects'][CLASS_IDS_REVERSED[idx]]['masks'][id]
                    mask = np.array(Image.open(BytesIO(base64.b64decode(mask_b64))))
                    area = data['objects'][CLASS_IDS_REVERSED[idx]]['area'][id]
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
