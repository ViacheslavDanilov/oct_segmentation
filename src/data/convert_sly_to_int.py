import json
import logging
import os
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

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# TODO: implement mask_processor
def get_mask_properties(
    figure: dict,
    mask: np.ndarray,
    crop: List[List[int]],
) -> Tuple[str, Polygon, List[List[Any]]]:
    if figure['geometryType'] == 'polygon':
        polygon = figure['geometry']['points']['exterior']
        cv2.fillPoly(mask, np.array([polygon]), 1)
        mask = mask[
            crop[0][1] : crop[1][1],
            crop[0][0] : crop[1][0],
        ]
        mask = mask.astype(bool)
    elif figure['geometryType'] == 'bitmap':
        mask = mask.astype(bool)
        bitmap = figure['geometry']['bitmap']['data']
        mask_ = sly.Bitmap.base64_2_data(bitmap)
        mask[
            figure['geometry']['bitmap']['origin'][1] : figure['geometry']['bitmap']['origin'][1]
            + mask_.shape[0],
            figure['geometry']['bitmap']['origin'][0] : figure['geometry']['bitmap']['origin'][0]
            + mask_.shape[1],
        ] = mask_[:, :]
    else:
        return None, None, None

    mask = mask[
        crop[0][1] : crop[1][1],
        crop[0][0] : crop[1][0],
    ]
    encoded_mask = sly.Bitmap.data_2_base64(mask)
    mask = sly.Bitmap(mask)
    contour = mask.to_contours()[0]
    bbox = [
        [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
        [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
    ]

    return encoded_mask, contour, bbox


def process_single_annotation(
    dataset: sly.VideoDataset,
    class_ids: dict,
    crop: List[List[int]],
    img_dir: str,
) -> pd.DataFrame:
    df_ann = pd.DataFrame()
    study = dataset.name
    for video_name in dataset:
        series = video_name.split('_')[1]
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
                'image_path': os.path.join(img_dir, img_name),
                'image_name': img_name,
                'study': study,
                'series': series,
                'slice': slice,
                'image_width': crop[1][0] - crop[0][0],
                'image_height': crop[1][1] - crop[0][1],
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
                'mask': None,
            }

            if len(ann_frame) != 0:
                for figure in ann_frame.figures.tolist()[0]:
                    # Extract figure features
                    obj = objects[objects['key'] == figure['objectKey']]
                    class_title = obj.classTitle.values[0]
                    mask = np.zeros((ann['size']['width'], ann['size']['height']))
                    encoded_mask, contour, bbox = get_mask_properties(
                        figure=figure,
                        mask=mask,
                        crop=crop,
                    )
                    if encoded_mask is None:
                        break

                    # Fill the result dictionary with the figure properties
                    result_dict['class_id'] = class_ids[class_title]
                    result_dict['class_name'] = class_title
                    result_dict['x1'] = bbox[0][0]
                    result_dict['y1'] = bbox[0][1]
                    result_dict['x2'] = bbox[1][0]
                    result_dict['y2'] = bbox[1][1]
                    result_dict['xc'] = int(np.mean([bbox[0][0], bbox[1][0]]))
                    result_dict['yc'] = int(np.mean([bbox[0][1], bbox[1][1]]))
                    result_dict['box_width'] = bbox[1][0] - bbox[0][0] + 1
                    result_dict['box_height'] = bbox[1][1] - bbox[0][1] + 1
                    result_dict['area'] = int(contour.area)
                    result_dict['mask'] = encoded_mask
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
        series = video_name.split('_')[1]
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
    save_dir: str,
) -> None:
    df = pd.concat(df_list)
    df.sort_values(['image_path', 'class_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='id',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    meta = json.load(open(os.path.join(cfg.data_dir, 'meta.json')))
    project_sly = sly.VideoProject(cfg.data_dir, sly.OpenMode.READ)
    class_ids = {value['title']: id + 1 for (id, value) in enumerate(meta['classes'])}
    img_dir = os.path.join(cfg.save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Process video
    Parallel(n_jobs=-1)(
        delayed(process_single_video)(
            dataset=dataset,
            src_dir=cfg.data_dir,
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
            class_ids=class_ids,
            crop=cfg.crop,
        )
        for dataset in tqdm(project_sly.datasets, desc='Process annotations')
    )

    # Save annotation metadata
    save_metadata(
        df_list=df_list,
        save_dir=cfg.save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
