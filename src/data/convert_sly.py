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

annotation_write = {
    'Image path': None,
    'Image name': None,
    'Study': None,
    'Series': None,
    'Slice': None,
    'Image width': None,
    'Image height': None,
    'Class ID': None,
    'Class': None,
    'x1': None,
    'y1': None,
    'x2': None,
    'y2': None,
    'xc': None,
    'yc': None,
    'Box width': None,
    'Box height': None,
    'Area': None,
    'Mask': None,
}


def get_feature_mask(
    figure: dict,
    mask: np.ndarray,
    crop: List[List[int]],
):
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


def parse_single_annotation(
    dataset: sly.VideoDataset,
    img_dir: str,
    fields: List[str],
    class_ids: dict,
    crop: List[List[int]],
    return_annotation: bool,
    save_annotation_path: str,
):

    if return_annotation:
        annotation = pd.DataFrame()
    else:
        annotation = None

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

            # initialization result dict
            result_dict = annotation_write.copy()
            result_dict['Image path'] = os.path.join(img_dir, img_name)
            result_dict['Image name'] = img_name
            result_dict['Study'] = study
            result_dict['Series'] = series
            result_dict['Slice'] = slice
            result_dict['Image width'] = crop[1][0] - crop[0][0]
            result_dict['Image height'] = crop[1][1] - crop[0][1]

            if len(ann_frame) != 0:
                for figure in ann_frame.figures.tolist()[0]:
                    # GET feature figure
                    obj = objects[objects['key'] == figure['objectKey']]
                    class_title = obj.classTitle.values[0]
                    mask = np.zeros((ann['size']['width'], ann['size']['height']))
                    encoded_mask, contour, bbox = get_feature_mask(
                        figure=figure,
                        mask=mask,
                        crop=crop,
                    )
                    if encoded_mask is None:
                        break

                    # result dict add feature values
                    result_dict['Class ID'] = class_ids[class_title]
                    result_dict['Class'] = class_title
                    result_dict['x1'] = bbox[0][0]
                    result_dict['y1'] = bbox[0][1]
                    result_dict['x2'] = bbox[1][0]
                    result_dict['y2'] = bbox[1][1]
                    result_dict['xc'] = int(np.mean([bbox[0][0], bbox[1][0]]))
                    result_dict['yc'] = int(np.mean([bbox[0][1], bbox[1][1]]))
                    result_dict['Box width'] = bbox[1][0] - bbox[0][0]
                    result_dict['Box height'] = bbox[1][1] - bbox[0][1]
                    result_dict['Area'] = int(contour.area)
                    result_dict['Mask'] = encoded_mask

                    # save annotation
                    if save_annotation_path:
                        with open(save_annotation_path, 'a', newline='') as f:
                            writer = DictWriter(f, fieldnames=fields)
                            writer.writerow(
                                result_dict,
                            )
                            f.close()
                    if return_annotation:
                        annotation = pd.concat([annotation, pd.DataFrame(result_dict, index=[0])])

            else:
                # IF ann none save empty annotation
                if save_annotation_path:
                    with open(save_annotation_path, 'a', newline='') as f:
                        writer = DictWriter(f, fieldnames=fields)
                        writer.writerow(
                            result_dict,
                        )
                        f.close()
                if return_annotation:
                    annotation = pd.concat([annotation, pd.DataFrame(result_dict, index=[0])])

    return annotation


def parse_single_video(
    dataset: sly.VideoDataset,
    src_dir: str,
    img_dir: str,
    crop: List[List[int]],
) -> None:
    study = dataset.name
    for video_name in dataset:
        series = video_name.split('_')[1]
        vid = cv2.VideoCapture(
            f'{src_dir}/{study}/{dataset.item_dir_name}/{video_name}',
        )
        idx = 1
        while True:
            _, img = vid.read()
            if _:
                img = img[
                    crop[0][1] : crop[1][1],
                    crop[0][0] : crop[1][0],
                    :,
                ]
                img_name = f'{study}_{series}_{idx:03d}.png'
                cv2.imwrite(os.path.join(img_dir, img_name), img)
                idx += 1
            else:
                break


def annotation_parsing(
    img_dir,
    datasets,
    class_ids,
    fields: List[str],
    crop: List[List[int]],
    save_annotation_path: str = None,
    return_annotation: bool = True,
):
    if save_annotation_path is not None:
        with open(save_annotation_path, 'w', newline='') as f_object:
            writer = DictWriter(f_object, fieldnames=fields)
            writer.writeheader()
            f_object.close()

    num_cores = multiprocessing.cpu_count()
    annotation = Parallel(n_jobs=num_cores, backend='threading')(
        delayed(parse_single_annotation)(
            dataset=dataset,
            img_dir=img_dir,
            fields=fields,
            class_ids=class_ids,
            crop=crop,
            return_annotation=return_annotation,
            save_annotation_path=save_annotation_path,
        )
        for dataset in tqdm(datasets, desc='Annotation parsing')
    )
    if return_annotation:
        return annotation
    else:
        return None


def video_parsing(
    datasets,
    img_dir: str,
    src_dir: str,
    crop: List[List[int]],
) -> None:
    os.makedirs(img_dir, exist_ok=True)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(parse_single_video)(
            dataset=dataset,
            src_dir=src_dir,
            img_dir=img_dir,
            crop=crop,
        )
        for dataset in tqdm(datasets, desc='Video parsing')
    )


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(
    cfg: DictConfig,
) -> None:
    meta = json.load(open(os.path.join(cfg.sly_to_int.study_dir, 'meta.json')))
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
    class_ids = {value['title']: id for (id, value) in enumerate(meta['classes'])}
    img_dir = os.path.join(cfg.sly_to_int.save_dir, 'img')

    # 1. Video parsing
    # video_parsing(
    #     datasets=project_sly.datasets,
    #     img_dir=img_dir,
    #     src_dir=cfg.sly_to_int.study_dir,
    #     crop=cfg.sly_to_int.crop,
    # )

    # 2. Annotation parsing
    annotation = annotation_parsing(
        img_dir=img_dir,
        datasets=project_sly.datasets,
        class_ids=class_ids,
        fields=fields,
        crop=cfg.sly_to_int.crop,
        save_annotation_path=os.path.join(cfg.sly_to_int.save_dir, 'metadata.csv'),
        return_annotation=True,
    )

    # 3. Save annotation .xlsx
    if annotation is not None:
        df = pd.DataFrame()
        for ann in annotation:
            df = pd.concat([df, ann])
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


if __name__ == '__main__':
    main()
