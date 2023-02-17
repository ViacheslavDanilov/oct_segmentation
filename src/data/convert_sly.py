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


def append_annotation(
    img_name: str,
    img_dir: str,
    study: str,
    series: str,
    slice: str,
    crop: List[List[int]],
    class_ids: dict,
    class_title: str,
    ann_path: str,
    fields: List[str],
    bbox: List[List[int]],
    area: int,
    encoded_mask: str,
) -> None:
    with open(ann_path, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writerow(
            {
                'Image path': os.path.join(img_dir, img_name),
                'Image name': img_name,
                'Study': study,
                'Series': series,
                'Slice': slice,
                'Image width': crop[1][0] - crop[0][0],
                'Image height': crop[1][1] - crop[0][1],
                'Class ID': class_ids[class_title],
                'Class': class_title,
                'x1': bbox[0][0],
                'y1': bbox[0][1],
                'x2': bbox[1][0],
                'y2': bbox[1][1],
                'xc': int(np.mean([bbox[0][0], bbox[1][0]])),
                'yc': int(np.mean([bbox[0][1], bbox[1][1]])),
                'Box width': bbox[1][0] - bbox[0][0],
                'Box height': bbox[1][1] - bbox[0][1],
                'Area': area,
                'Mask': encoded_mask,
            },
        )
        f.close()


def process_frame(
    img: np.ndarray,
    crop: List[List[int]],
    study: str,
    series: str,
    ann: dict,
    frame: dict,
    fields: List[str],
    class_ids: dict,
    objects: pd.DataFrame,
    ann_path: str,
    img_dir: str,
) -> None:
    img = img[
        crop[0][1] : crop[1][1],
        crop[0][0] : crop[1][0],
        :,
    ]
    slice = f'{frame["index"] + 1:03d}'
    img_name = f'{study}_{series}_{slice}.png'
    cv2.imwrite(os.path.join(img_dir, img_name), img)

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
        encoded_mask = sly.Bitmap.data_2_base64(mask)
        mask = sly.Bitmap(mask)
        contour = mask.to_contours()[0]
        bbox = [
            [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
            [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
        ]

        append_annotation(
            crop=crop,
            ann_path=ann_path,
            fields=fields,
            img_dir=img_dir,
            img_name=img_name,
            study=study,
            series=series,
            slice=slice,
            class_ids=class_ids,
            class_title=class_title,
            bbox=bbox,
            area=int(contour.area),
            encoded_mask=encoded_mask,
        )


def process_study(
    dataset: sly.VideoDataset,
    src_dir: str,
    fields: List[str],
    class_ids: dict,
    ann_path: str,
    img_dir: str,
    crop: List[List[int]],
) -> None:
    study = dataset.name
    for video_name in dataset:
        series = video_name.split('_')[1]
        vid = cv2.VideoCapture(
            f'{src_dir}/{study}/{dataset.item_dir_name}/{video_name}',
        )
        ann = json.load(
            open(
                f'{src_dir}/{study}/{dataset.ann_dir_name}/{video_name}.json',
            ),
        )
        objects = pd.DataFrame(ann['objects'])
        for frame in ann['frames']:
            vid.set(1, frame['index'])
            _, img = vid.read()
            process_frame(
                img=img,
                crop=crop,
                study=study,
                series=series,
                ann=ann,
                frame=frame,
                fields=fields,
                objects=objects,
                class_ids=class_ids,
                ann_path=ann_path,
                img_dir=img_dir,
            )


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(
    cfg: DictConfig,
) -> None:
    project_sly = sly.VideoProject(cfg.sly_to_int.study_dir, sly.OpenMode.READ)
    fields = [
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
    class_ids = {value['title']: id for (id, value) in enumerate(meta['classes'])}

    ann_path = os.path.join(cfg.sly_to_int.save_dir, 'metadata.csv')
    img_dir = os.path.join(cfg.sly_to_int.save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_path, 'w', newline='') as f_object:
        writer = DictWriter(f_object, fieldnames=fields)
        writer.writeheader()
        f_object.close()

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(process_study)(
            dataset=dataset,
            src_dir=cfg.sly_to_int.study_dir,
            fields=fields,
            class_ids=class_ids,
            ann_path=ann_path,
            img_dir=img_dir,
            crop=cfg.sly_to_int.crop,
        )
        for dataset in tqdm(project_sly, desc='Dataset conversion')
    )

    df = pd.read_csv(ann_path)
    df.sort_values(['Image path', 'Class ID'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(cfg.sly_to_int.save_dir, 'metadata.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )
    os.remove(ann_path)


if __name__ == '__main__':
    main()
