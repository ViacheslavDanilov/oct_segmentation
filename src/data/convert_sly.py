import json
import multiprocessing
import os
from csv import DictWriter
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

# global param
id = 0


def annotation_write(
    crop: List[List[int]],
    annotation_path: str,
    fieldnames: List[str],
    images_dir: str,
    img_name: str,
    patient_name: str,
    series: int,
    slice: str,
    classes_id: dict,
    class_title: str,
    rectangle: List[List[int]],
    area: float,
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
                'Image width': crop[1][0] - crop[0][0],
                'Image height': crop[1][1] - crop[0][1],
                'Class ID': classes_id[class_title],
                'Class': class_title,
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
    crop: List[List[int]],
    classes_id: dict,
    img: np.ndarray,
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
        crop[0][1] : crop[1][1],
        crop[0][0] : crop[1][0],
        :,
    ]
    slice = f'{frame["index"] + 1:03d}'
    img_name = f'{patient_name}_{series:1d}_{slice}.png'
    cv2.imwrite(os.path.join(images_dir, img_name), img)
    # Annotation
    for figure in frame['figures']:
        obj = objects[objects['key'] == figure['objectKey']]
        class_title = obj.classTitle.values[0]
        mask = np.zeros((ann['size']['width'], ann['size']['height']))

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
            crop[0][1] : crop[1][1],
            crop[0][0] : crop[1][0],
        ]
        encoded_string = sly.Bitmap.data_2_base64(mask)
        mask = sly.Bitmap(mask)
        contour = mask.to_contours()[0]
        rectangle = [
            [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
            [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
        ]

        annotation_write(
            crop=crop,
            annotation_path=annotation_path,
            fieldnames=fieldnames,
            images_dir=images_dir,
            img_name=img_name,
            patient_name=patient_name,
            series=series,
            slice=slice,
            classes_id=classes_id,
            class_title=class_title,
            rectangle=rectangle,
            area=contour.area,
            encoded_string=encoded_string,
        )


def _processing_item(
    study_dir: str,
    fieldnames: List[str],
    dataset_fs: sly.VideoDataset,
    classes_id: dict,
    images_dir: str,
    annotation_path: str,
    crop: List[List[int]],
):
    patient_name = dataset_fs.name
    for series, item_name in enumerate(dataset_fs):
        series += 1
        vid = cv2.VideoCapture(
            f'{study_dir}/{patient_name}/{dataset_fs.item_dir_name}/{item_name}',
        )
        ann = json.load(
            open(
                f'{study_dir}/{patient_name}/{dataset_fs.ann_dir_name}/{item_name}.json',
            ),
        )
        objects = pd.DataFrame(ann['objects'])
        for frame in ann['frames']:
            vid.set(1, frame['index'])
            _, img = vid.read()
            _processing_frame(
                crop=crop,
                classes_id=classes_id,
                img=np.array(img),
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
            cfg.sly_to_int.study_dir,
            fieldnames,
            dataset_fs,
            classes_id,
            images_dir,
            annotation_path,
            cfg.sly_to_int.crop,
        )
        for dataset_fs in tqdm(project_fs, desc='patients analysis')
    )
    pd.read_csv(annotation_path).to_excel(f'{os.path.splitext(annotation_path)[0]}.xlsx')


if __name__ == '__main__':
    main()
