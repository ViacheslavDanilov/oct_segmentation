import json
import logging
import os
import re
from typing import Any, List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from supervisely import Polygon
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_IDS

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def polygon_to_mask(
    polygon: List,
) -> Tuple[int, int, np.ndarray]:
    # Find the top-left and bottom-right corners of the bounding box
    x_min = min(vertex[0] for vertex in polygon)
    x_max = max(vertex[0] for vertex in polygon)
    y_min = min(vertex[1] for vertex in polygon)
    y_max = max(vertex[1] for vertex in polygon)

    # Compute mask height and width
    mask_height = y_max - y_min
    mask_width = x_max - x_min

    # Create a mask using polygon vertices
    polygon_np = np.array(polygon, dtype=np.int32)
    points = [polygon_np - (x_min, y_min)]
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    cv2.fillPoly(mask, pts=points, color=1)

    return x_min, y_min, mask


def bitmap_to_mask(
    encoded_bitmap: str,
) -> np.ndarray:
    mask = sly.Bitmap.base64_2_data(encoded_bitmap)

    return mask


def process_polygon_geometry(
    polygon: dict,
) -> Tuple[int, int, np.ndarray]:
    x_min, y_min, obj_mask = polygon_to_mask(polygon['exterior'])
    return x_min, y_min, obj_mask


def process_bitmap_geometry(
    bitmap: dict,
) -> Tuple[int, int, np.ndarray]:
    x_min, y_min = bitmap['origin']
    obj_mask = bitmap_to_mask(bitmap['data'])
    return x_min, y_min, obj_mask


def get_mask_properties(
    figure: dict,
    mask: np.ndarray,
    crop: List[List[int]],
) -> Tuple[str, Polygon, List[List[Any]]]:
    if figure['geometryType'] == 'polygon':
        x_min, y_min, obj_mask = process_polygon_geometry(figure['geometry']['points'])
    elif figure['geometryType'] == 'bitmap':
        x_min, y_min, obj_mask = process_bitmap_geometry(figure['geometry']['bitmap'])
    else:
        return None, None, None

    # Paste the mask of the object into the source mask
    mask[y_min : y_min + obj_mask.shape[0], x_min : x_min + obj_mask.shape[1]] = obj_mask[:, :]

    # Crop the resulting mask using the specified coordinates
    mask = mask[
        crop[0][1] : crop[1][1],
        crop[0][0] : crop[1][0],
    ]

    # Extract final information from the mask
    encoded_mask = sly.Bitmap.data_2_base64(mask)
    mask = sly.Bitmap(mask)
    contour = mask.to_contours()[0]
    bbox = [
        [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
        [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
    ]

    return encoded_mask, contour, bbox


def get_series_id(
    filename: str,
) -> int:
    # Extract the value between '_' and '.mp4'
    match = re.search(r'_(\d+)\.mp4', filename)

    if match:
        series_id = int(match.group(1))
    else:
        raise ValueError('No match found')

    return series_id


def process_single_annotation(
    dataset: sly.VideoDataset,
    img_dir: str,
    class_ids: dict,
    crop: List[List[int]],
) -> pd.DataFrame:
    df_ann = pd.DataFrame()
    study = dataset.name
    for video_name in dataset:
        series = get_series_id(video_name)
        ann_path = os.path.join(dataset.ann_dir, f'{video_name}.json')
        ann = json.load(open(ann_path))
        ann_frames = pd.DataFrame(ann['frames'])
        objects = pd.DataFrame(ann['objects'])
        for idx in range(ann['framesCount']):
            slice = f'{idx + 1:03d}'
            img_name = f'{study}_{series}_{slice}.png'
            if len(ann_frames) > 0:
                ann_frame = ann_frames.loc[ann_frames['index'] == idx]
            else:
                ann_frame = []

            # Initializing the dictionary with annotations
            result_dict = {
                'img_path': os.path.join(img_dir, img_name),
                'img_name': img_name,
                'study': study,
                'series': series,
                'slice': slice,
                'img_width': crop[1][0] - crop[0][0],
                'img_height': crop[1][1] - crop[0][1],
                'type': None,
                'class_id': None,
                'class_name': None,
                'x1': None,
                'y1': None,
                'x2': None,
                'y2': None,
                'xc': None,
                'yc': None,
                'box_width': None,
                'box_height': None,
                'area': None,
                'encoded_mask': None,
            }

            if len(ann_frame) != 0:
                for figure in ann_frame.figures.tolist()[0]:
                    # Extract figure features
                    obj = objects[objects['key'] == figure['objectKey']]
                    class_name = obj.classTitle.values[0]
                    mask = np.zeros((ann['size']['height'], ann['size']['width']))
                    encoded_mask, contour, bbox = get_mask_properties(
                        figure=figure,
                        mask=mask,
                        crop=crop,
                    )
                    if encoded_mask is None:
                        break

                    # Fill the result dictionary with the figure properties
                    result_dict['type'] = figure['geometryType']
                    result_dict['class_id'] = class_ids[class_name]
                    result_dict['class_name'] = class_name
                    result_dict['x1'] = bbox[0][0]
                    result_dict['y1'] = bbox[0][1]
                    result_dict['x2'] = bbox[1][0]
                    result_dict['y2'] = bbox[1][1]
                    result_dict['xc'] = int(np.mean([bbox[0][0], bbox[1][0]]))
                    result_dict['yc'] = int(np.mean([bbox[0][1], bbox[1][1]]))
                    result_dict['box_width'] = bbox[1][0] - bbox[0][0] + 1
                    result_dict['box_height'] = bbox[1][1] - bbox[0][1] + 1
                    result_dict['area'] = int(contour.area)
                    result_dict['encoded_mask'] = encoded_mask
                    df_ann = pd.concat([df_ann, pd.DataFrame(result_dict, index=[0])])

            # Save empty annotation if ann is None
            else:
                df_ann = pd.concat([df_ann, pd.DataFrame(result_dict, index=[0])])

    return df_ann


def process_single_video(
    dataset: sly.VideoDataset,
    src_dir: str,
    img_dir: str,
    crop: List[List[int]],
) -> None:
    study = dataset.name
    for video_name in dataset:
        series = get_series_id(video_name)
        video_path = os.path.join(src_dir, study, dataset.item_dir_name, video_name)
        vid = cv2.VideoCapture(video_path)

        idx = 1
        while True:
            _, img = vid.read()
            if _:
                img = img[crop[0][1] : crop[1][1], crop[0][0] : crop[1][0], :]
                img_name = f'{study}_{series}_{idx:03d}.png'
                img_path = os.path.join(img_dir, img_name)
                cv2.imwrite(img_path, img)
                idx += 1
            else:
                break

        vid.release()


def save_metadata(
    df_list: sly.Project.DatasetDict,
    project_dir: str,
    save_dir: str,
) -> None:
    df = pd.concat(df_list)
    df.sort_values(['img_path', 'class_id'], inplace=True)
    df['img_path'] = df['img_path'].apply(lambda x: os.path.relpath(x, project_dir))
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(save_dir, 'metadata.csv')
    df.to_csv(save_path, index_label='id')


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # Read project
    project_sly = sly.VideoProject(data_dir, sly.OpenMode.READ)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Process video
    Parallel(n_jobs=-1)(
        delayed(process_single_video)(
            dataset=dataset,
            src_dir=data_dir,
            img_dir=img_dir,
            crop=cfg.crop,
        )
        for dataset in tqdm(project_sly.datasets, desc='Process video')
    )

    # Process annotations
    df_list = Parallel(n_jobs=-1)(
        delayed(process_single_annotation)(
            dataset=dataset,
            img_dir=img_dir,
            class_ids=CLASS_IDS,
            crop=cfg.crop,
        )
        for dataset in tqdm(project_sly.datasets, desc='Process annotations')
    )

    # Save annotation metadata
    save_metadata(
        df_list=df_list,
        project_dir=PROJECT_DIR,
        save_dir=save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
