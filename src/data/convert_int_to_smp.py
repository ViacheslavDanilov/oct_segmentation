import multiprocessing
import os
import shutil
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_mask(
        img_path: str,
        classes: List[str],
        data: pd.DataFrame,
        save_dir: str,
) -> None:
    if len(data) > 0 and len(list(set(classes) & set(data.class_name.unique()))) > 0:
        mask = np.zeros((int(data.image_width.mean()), int(data.image_height.mean())))
        for _, row in data.iterrows():
            if row.class_name in classes:
                figure_data = sly.Bitmap.base64_2_data(row.mask_b64)
                mask[figure_data == True] = row.class_id + 1
        cv2.imwrite(f'{save_dir}/ann/{os.path.basename(img_path)}', mask)
        shutil.copy(img_path, f'{save_dir}/img/{os.path.basename(img_path)}')


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='convert_int_to_smp',
    version_base=None,
)
def main(
        cfg: DictConfig,
):
    df = pd.read_excel(cfg.df_path)
    if not os.path.exists(cfg.save_dir):
        os.makedirs(f'{cfg.save_dir}/train/img')
        os.makedirs(f'{cfg.save_dir}/train/ann')
        os.makedirs(f'{cfg.save_dir}/val/img')
        os.makedirs(f'{cfg.save_dir}/val/ann')

    num_cores = multiprocessing.cpu_count()

    images_path = df.image_path.unique()
    train_images_path, val_images_path = train_test_split(images_path, train_size=cfg.train, test_size=cfg.test,
                                                          shuffle=True, random_state=11)

    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df['image_path'] == img_path],
            save_dir=f'{cfg.save_dir}/train'
        )
        for img_path in tqdm(train_images_path, desc='training images analysis')
    )
    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df['image_path'] == img_path],
            save_dir=f'{cfg.save_dir}/val'
        )
        for img_path in tqdm(val_images_path, desc='validation images analysis')
    )


if __name__ == '__main__':
    main()
