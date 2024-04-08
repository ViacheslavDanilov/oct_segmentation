import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_COLOR, CLASS_ID, convert_base64_to_numpy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_data_directories(
    num_folds: int,
    save_dir: str,
) -> None:
    for fold_idx in range(1, num_folds + 1):
        fold_path = Path(save_dir) / f'fold_{fold_idx}'
        for subset in ['train', 'test']:
            for dir_type in ['img', 'mask', 'mask_color']:
                (fold_path / subset / dir_type).mkdir(parents=True, exist_ok=True)


def process_metadata(
    df: pd.DataFrame,
    class_names: List[str] = None,
) -> pd.DataFrame:
    """Extract additional meta.

    Args:
        df: path to directory containing images and metadata
        class_names: a list of classes to include in the dataset
    Returns:
        df: data frame derived from a meta file
    """

    if class_names is not None:
        df = df[df['class_name'].isin(class_names)]

    df = df.dropna(subset=['class_name'])

    assert len(df) > 0, 'All items have been excluded or dropped'

    return df


def cross_validation_split(
    df: pd.DataFrame,
    id_column: str,
    num_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ids = df[id_column].unique()
    kf = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    splits = []
    for train_idx, test_idx in kf.split(ids):
        train_ids = ids[train_idx]
        test_ids = ids[test_idx]
        df_train = df[df[id_column].isin(train_ids)]
        df_test = df[df[id_column].isin(test_ids)]
        splits.append((df_train, df_test))

    return splits


# def split_dataset(
#     df: pd.DataFrame,
#     train_size: float = 0.80,
#     seed: int = 11,
# ) -> pd.DataFrame:
#     # Split dataset by studies
#     df_unique_studies = np.unique(df.study.values)
#     train_studies, test_studies = train_test_split(
#         df_unique_studies,
#         train_size=train_size,
#         shuffle=True,
#         random_state=seed,
#     )
#
#     # Extract training and testing subsets by indexes
#     df_train = df[df['study'].isin(train_studies)]
#     df_test = df[df['study'].isin(test_studies)]
#     df_train = df_train.assign(split='train')
#     df_test = df_test.assign(split='test')
#
#     # Get list of train and test paths
#     log.info('Split..........: Studies / Images')
#     log.info(f'Train..........: {len(df_train["study"].unique())} / {len(df_train)}')
#     log.info(f'Test images....: {len(df_test["study"].unique())} / {len(df_test)}')
#
#     df_out = pd.concat([df_train, df_test])
#
#     return df_out


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
    save_path = os.path.join(save_dir, 'metadata.csv')
    df.to_csv(save_path, index_label='id')


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

    # Create directories for storing images and masks
    create_data_directories(
        num_folds=cfg.num_folds,
        save_dir=save_dir,
    )

    # Read and process data frame
    csv_path = os.path.join(data_dir, 'metadata.csv')
    df = pd.read_csv(csv_path)
    df_filtered = process_metadata(
        df=df,
        class_names=cfg.class_names,
    )

    # Cross-validation split of the dataset
    splits = cross_validation_split(
        df=df_filtered,
        id_column=cfg.split_column,
        num_folds=cfg.num_folds,
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
