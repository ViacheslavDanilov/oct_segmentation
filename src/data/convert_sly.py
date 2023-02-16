import json
import multiprocessing
import os
from csv import DictWriter
from typing import List

import cv2
import hydra
import numpy as np
import pandas
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

# global param
id = 0


def annotation_write(
    cfg: DictConfig,
    annotation_path: str,
    fieldnames: List[str],
    images_dir: str,
    img_name: str,
    patient_name: str,
    series: int,
    classes_id: dict,
    classTitle: str,
    rectangle: List[List[int]],
    area: int,
    encoded_string: str,
):
    global id
    with open(annotation_path, 'a', newline='') as f_object:
        writer = DictWriter(f_object, fieldnames=fieldnames)
        writer.writerow(
            {
                'ID': id,
                'Image path': os.path.join(images_dir, img_name),
                'Image name': img_name,
                'Study': patient_name,
                'Series': series,
                'Slice': slice,
                'Image width': cfg.sly_to_int.crop[1][0] - cfg.sly_to_int.crop[0][0],
                'Image height': cfg.sly_to_int.crop[1][1] - cfg.sly_to_int.crop[0][1],
                'Class ID': classes_id[classTitle],
                'Class': classTitle,
                'x1': rectangle[0][0],
                'y1': rectangle[0][1],
                'x2': rectangle[1][0],
                'y2': rectangle[1][1],
                'xc': int(np.mean([rectangle[0][0], rectangle[1][0]])),
                'yc': int(np.mean([rectangle[0][1], rectangle[1][1]])),
                'Box width': rectangle[1][0] - rectangle[0][0],
                'Box height': rectangle[1][1] - rectangle[0][1],
                'Area': area,
                'Mask': encoded_string,
            },
        )
        id += 1
        f_object.close()


def _processing_frame(
    cfg: DictConfig,
    classes_id: dict,
    img: List[List[float]],
    frame: dict,
    fieldnames: List[str],
    ann: dict,
    objects: pd.DataFrame,
    patient_name: str,
    series: int,
    images_dir: str,
    annotation_path: str,
):
    img = img[
        cfg.sly_to_int.crop[0][1] : cfg.sly_to_int.crop[1][1],
        cfg.sly_to_int.crop[0][0] : cfg.sly_to_int.crop[1][0],
        :,
    ]
    img_name = f'{patient_name}_{series:1d}_{frame["index"]:03d}.png'
    cv2.imwrite(os.path.join(images_dir, img_name), img)
    # Annotation
    for figure in frame['figures']:
        obj = objects[objects['key'] == figure['objectKey']]
        classTitle = obj.classTitle.values[0]
        mask = np.zeros((ann['size']['width'], ann['size']['height']))

        if figure['geometryType'] == 'polygon':
            polygon = figure['geometry']['points']['exterior']
            cv2.fillPoly(mask, np.array([polygon]), 1)
            mask = mask[
                cfg.sly_to_int.crop[0][1] : cfg.sly_to_int.crop[1][1],
                cfg.sly_to_int.crop[0][0] : cfg.sly_to_int.crop[1][0],
            ]
            mask = mask.astype(bool)
        elif figure['geometryType'] == 'bitmap':
            mask = mask.astype(bool)
            bitmap = figure['geometry']['bitmap']['data']
            mask_ = sly.Bitmap.base64_2_data(bitmap)
            mask[
                figure['geometry']['bitmap']['origin'][1] : figure['geometry']['bitmap']['origin'][
                    1
                ]
                + mask_.shape[0],
                figure['geometry']['bitmap']['origin'][0] : figure['geometry']['bitmap']['origin'][
                    0
                ]
                + mask_.shape[1],
            ] = mask_[:, :]
        else:
            break

        mask = mask[
            cfg.sly_to_int.crop[0][1] : cfg.sly_to_int.crop[1][1],
            cfg.sly_to_int.crop[0][0] : cfg.sly_to_int.crop[1][0],
        ]

        encoded_string = sly.Bitmap.data_2_base64(mask)
        n_m = np.nonzero(mask)
        area = len(n_m[0])
        rectangle = [
            [min(n_m[1]), min(n_m[0])],
            [max(n_m[1]), max(n_m[0])],
        ]

        annotation_write(
            cfg=cfg,
            annotation_path=annotation_path,
            fieldnames=fieldnames,
            images_dir=images_dir,
            img_name=img_name,
            patient_name=patient_name,
            series=series,
            classes_id=classes_id,
            classTitle=classTitle,
            rectangle=rectangle,
            area=area,
            encoded_string=encoded_string,
        )


def _processing_item(
    cfg: DictConfig,
    fieldnames: List[str],
    dataset_fs: sly.VideoDataset,
    classes_id: dict,
    images_dir: str,
    annotation_path: str,
):
    patient_name = dataset_fs.name
    for series, item_name in enumerate(dataset_fs):
        series += 1
        vid = cv2.VideoCapture(
            f'{cfg.sly_to_int.study_dir}/{patient_name}/{dataset_fs.item_dir_name}/{item_name}',
        )
        ann = json.load(
            open(
                f'{cfg.sly_to_int.study_dir}/{patient_name}/{dataset_fs.ann_dir_name}/{item_name}.json',
            ),
        )
        objects = pandas.DataFrame(ann['objects'])
        for frame in ann['frames']:
            vid.set(1, frame['index'])
            _, img = vid.read()
            _processing_frame(
                cfg=cfg,
                classes_id=classes_id,
                img=img,
                frame=frame,
                fieldnames=fieldnames,
                ann=ann,
                objects=objects,
                patient_name=patient_name,
                series=series,
                images_dir=images_dir,
                annotation_path=annotation_path,
            )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='data',
    version_base=None,
)
def main(
    cfg: DictConfig,
):
    project_fs = sly.VideoProject(cfg.sly_to_int.study_dir, sly.OpenMode.READ)
    annotation_path = os.path.join(cfg.sly_to_int.save_dir, 'annotation.csv')
    images_dir = os.path.join(cfg.sly_to_int.save_dir, 'img')
    fieldnames = [
        'ID',
        'Image path',
        'Image name',
        'Study',
        'Series',
        'Slice',
        'Image width',
        'Image height',
        'Class ID',
        'Class',
        'x1',
        'y1',
        'x2',
        'y2',
        'xc',
        'yc',
        'Box width',
        'Box height',
        'Area',
        'Mask',
    ]
    meta = json.load(open(os.path.join(cfg.sly_to_int.study_dir, 'meta.json')))
    classes_id = {value['title']: id for (id, value) in enumerate(meta['classes'])}

    if os.path.exists(annotation_path):
        os.remove(annotation_path)
    with open(annotation_path, 'a', newline='') as f_object:
        writer = DictWriter(f_object, fieldnames=fieldnames)
        writer.writeheader()
        f_object.close()

    os.makedirs(images_dir, exist_ok=True)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(_processing_item)(
            cfg,
            fieldnames,
            dataset_fs,
            classes_id,
            images_dir,
            annotation_path,
        )
        for dataset_fs in tqdm(project_fs, desc='patients analysis')
    )


if __name__ == '__main__':
    main()
