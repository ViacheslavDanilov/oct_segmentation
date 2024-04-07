import logging
import os
import shutil
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_COLOR, CLASS_ID, convert_base64_to_numpy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_metadata(
    data_dir: str,
    exclude_classes: List[str] = None,
) -> pd.DataFrame:
    """Extract additional meta.

    Args:
        data_dir: path to directory containing images and metadata
        exclude_classes: a list of classes to exclude from the dataset
    Returns:
        meta: data frame derived from a meta file
    """
    df_path = os.path.join(data_dir, 'metadata.xlsx')
    df = pd.read_excel(df_path)
    df.drop('id', axis=1, inplace=True)
    df = df[~df['class_name'].isin(exclude_classes)]
    df = df.dropna(subset=['class_name'])

    assert len(df) > 0, 'All items have been excluded or dropped'

    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.80,
    seed: int = 11,
) -> pd.DataFrame:
    # Split dataset by studies
    df_unique_studies = np.unique(df.study.values)
    train_studies, test_studies = train_test_split(
        df_unique_studies,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
    )

    # Extract training and testing subsets by indexes
    df_train = df[df['study'].isin(train_studies)]
    df_test = df[df['study'].isin(test_studies)]
    df_train = df_train.assign(split='train')
    df_test = df_test.assign(split='test')

    # Get list of train and test paths
    log.info('Split..........: Studies / Images')
    log.info(f'Train..........: {len(df_train["study"].unique())} / {len(df_train)}')
    log.info(f'Test images....: {len(df_test["study"].unique())} / {len(df_test)}')

    df_out = pd.concat([df_train, df_test])

    return df_out


def process_mask(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    if len(df) > 0:
        obj_ = df.iloc[0]
        img_name = os.path.basename(obj_.image_path)
        mask = np.zeros((obj_.image_height, obj_.image_width))
        mask_color = np.zeros((obj_.image_height, obj_.image_width, 3), dtype='uint8')
        mask_color[:, :] = (128, 128, 128)
        for _, obj in df.iterrows():
            obj_mask = convert_base64_to_numpy(obj.encoded_mask).astype('uint8')
            mask[obj_mask == 1] = CLASS_ID[obj.class_name]
            mask_color[mask == CLASS_ID[obj.class_name]] = CLASS_COLOR[obj.class_name]

        cv2.imwrite(f'{save_dir}/mask/{img_name}', mask)
        cv2.imwrite(f'{save_dir}/mask_color/{img_name}', mask_color)
        shutil.copy(obj_.image_path, f'{save_dir}/img/{img_name}')


def save_metadata(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    df['image_path'] = df.apply(
        func=lambda row: os.path.join(save_dir, row['split'], 'img', row['image_name']),
        axis=1,
    )
    df.sort_values(by=['image_path'], inplace=True)
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
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_int_to_cv',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = os.path.join(PROJECT_DIR, cfg.data_dir)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)

    for subset in ['train', 'test']:
        for dir_type in ['img', 'mask', 'mask_color']:
            os.makedirs(f'{save_dir}/{subset}/{dir_type}', exist_ok=True)

    # Read and process data frame
    df = process_metadata(
        data_dir=data_dir,
        exclude_classes=cfg.exclude_classes,
    )

    # Split dataset by studies
    df = split_dataset(
        df=df,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )

    # Process images and masks
    train_groups = df[df['split'] == 'train'].groupby('image_path')
    test_groups = df[df['split'] == 'test'].groupby('image_path')

    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            df=train_group,
            save_dir=os.path.join(save_dir, 'train'),
        )
        for _, train_group in tqdm(train_groups, desc='Preparation of training subset')
    )
    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            df=test_group,
            save_dir=os.path.join(save_dir, 'test'),
        )
        for _, test_group in tqdm(test_groups, desc='Preparation of testing subset')
    )

    # Save dataset metadata
    save_metadata(
        df=df,
        save_dir=save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
