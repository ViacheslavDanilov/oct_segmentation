import json
import os
from csv import DictWriter

import cv2
import hydra
import numpy as np
import pandas
import supervisely_lib as sly
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='data_sly',
    version_base=None,
)
def main(
    cfg: DictConfig,
):
    project_fs = sly.VideoProject(cfg.meta.study_dir, sly.OpenMode.READ)
    header_w = True
    annotation_path = os.path.join(cfg.meta.save_dir, 'annotation.csv')
    images_dir = os.path.join(cfg.meta.save_dir, 'img')
    fieldnames = [
        'ID',
        'Image path',
        'Image name',
        'Study name',
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
    meta = json.load(open(os.path.join(cfg.meta.study_dir, 'meta.json')))
    classes_id = {value['title']: id for id, value in enumerate(meta['classes'])}

    if os.path.exists(annotation_path):
        os.remove(annotation_path)
    os.makedirs(images_dir, exist_ok=True)

    for dataset_fs in tqdm(project_fs, desc='patients analysis'):
        patient_name = dataset_fs.name
        for series, item_name in enumerate(dataset_fs):
            series += 1
            vid = cv2.VideoCapture(
                f'{cfg.meta.study_dir}/{patient_name}/{dataset_fs.item_dir_name}/{item_name}',
            )
            ann = json.load(
                open(
                    f'{cfg.meta.study_dir}/{patient_name}/{dataset_fs.ann_dir_name}/{item_name}.json',
                ),
            )
            objects = pandas.DataFrame(ann['objects'])
            id = 0
            for frame in ann['frames']:
                # Image
                vid.set(1, frame['index'])
                _, img = vid.read()
                img = img[
                    cfg.meta.window_position[0][1] : cfg.meta.window_position[1][1],
                    cfg.meta.window_position[0][0] : cfg.meta.window_position[1][0],
                    :,
                ]
                slice = str(frame['index']).zfill(4)
                img_name = f'{patient_name}_{series}_{slice}.png'
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
                            cfg.meta.window_position[0][1] : cfg.meta.window_position[1][1],
                            cfg.meta.window_position[0][0] : cfg.meta.window_position[1][0],
                        ]
                        mask = mask.astype(bool)
                    elif figure['geometryType'] == 'bitmap':
                        mask = mask.astype(bool)
                        bitmap = figure['geometry']['bitmap']['data']
                        mask_ = sly.Bitmap.base64_2_data(bitmap)
                        mask[
                            figure['geometry']['bitmap']['origin'][1] : figure['geometry'][
                                'bitmap'
                            ]['origin'][1]
                            + mask_.shape[0],
                            figure['geometry']['bitmap']['origin'][0] : figure['geometry'][
                                'bitmap'
                            ]['origin'][0]
                            + mask_.shape[1],
                        ] = mask_[:, :]
                    else:
                        break

                    mask = mask[
                        cfg.meta.window_position[0][1] : cfg.meta.window_position[1][1],
                        cfg.meta.window_position[0][0] : cfg.meta.window_position[1][0],
                    ]

                    encoded_string = sly.Bitmap.data_2_base64(mask)
                    n_m = np.nonzero(mask)
                    area = len(n_m[0])
                    rectangle = [
                        [min(n_m[1]), min(n_m[0])],
                        [max(n_m[1]), max(n_m[0])],
                    ]

                    with open(annotation_path, 'a', newline='') as f_object:
                        writer = DictWriter(f_object, fieldnames=fieldnames)
                        if header_w:
                            writer.writeheader()
                            header_w = False
                        writer.writerow(
                            {
                                'ID': id,
                                'Image path': os.path.join(images_dir, img_name),
                                'Image name': img_name,
                                'Study name': patient_name,
                                'Series': series,
                                'Slice': slice,
                                'Image width': cfg.meta.window_position[1][0]
                                - cfg.meta.window_position[0][0],
                                'Image height': cfg.meta.window_position[1][1]
                                - cfg.meta.window_position[0][1],
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


if __name__ == '__main__':
    main()
