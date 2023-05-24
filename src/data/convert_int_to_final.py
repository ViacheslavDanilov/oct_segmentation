import logging
import os
import shutil
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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
                mask[figure_data is True] = row.class_id + 1
        cv2.imwrite(f'{save_dir}/mask/{os.path.basename(img_path)}', mask)
        shutil.copy(img_path, f'{save_dir}/img/{os.path.basename(img_path)}')


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    for subset in ['train', 'test']:
        for dir_type in ['img', 'mask']:
            os.makedirs(f'{cfg.save_dir}/{subset}/{dir_type}', exist_ok=True)

    df = pd.read_excel(cfg.df_path)
    studies = np.unique(df.study.values)
    train_studies, test_studies = train_test_split(
        studies,
        train_size=cfg.train_size,
        shuffle=True,
        random_state=cfg.seed,
    )

    train_img_paths = df[df['study'].isin(train_studies)].image_path.values
    test_img_paths = df[df['study'].isin(test_studies)].image_path.values
    log.info(f'Train images...: {len(train_img_paths)}')
    log.info(f'Test images....: {len(train_img_paths)}')

    Parallel(n_jobs=-1, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df['image_path'] == img_path],
            save_dir=f'{cfg.save_dir}/train',
        )
        for img_path in tqdm(train_img_paths, desc='Preparing the training subset')
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df['image_path'] == img_path],
            save_dir=f'{cfg.save_dir}/test',
        )
        for img_path in tqdm(test_img_paths, desc='Preparing the testing subset')
    )


if __name__ == '__main__':
    main()
