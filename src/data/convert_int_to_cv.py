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
from src.data.mask_processor import MaskProcessor
from src.data.utils import CLASS_COLOR_BGR, CLASS_ID, convert_base64_to_numpy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_data_directories(
    num_folds: int,
    save_dir: str,
    save_color_mask: bool = True,
) -> None:
    dir_types = ['img', 'mask', 'mask_color'] if save_color_mask else ['img', 'mask']
    for fold_idx in range(1, num_folds + 1):
        fold_path = Path(save_dir) / f'fold_{fold_idx}'
        for subset in ['train', 'test']:
            for dir_type in dir_types:
                (fold_path / subset / dir_type).mkdir(parents=True, exist_ok=True)


def process_metadata(
    df: pd.DataFrame,
    class_names: List[str] = None,
) -> pd.DataFrame:
    if class_names is not None:
        df = df[df['class_name'].isin(class_names)]

    df = df.dropna(subset=['class_name'])

    assert len(df) > 0, 'All items have been excluded or dropped'

    return df


def update_metadata(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fold_idx: int,
) -> pd.DataFrame:
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train.loc[:, 'split'] = 'train'
    df_test.loc[:, 'split'] = 'test'
    df_train.loc[:, 'fold'] = fold_idx
    df_test.loc[:, 'fold'] = fold_idx

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.drop(columns=['id', 'encoded_mask', 'type'], inplace=True)

    df.sort_values(['img_name', 'class_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    return df


def cross_validation_split(
    df: pd.DataFrame,
    split_column: str,
    num_folds: int,
    seed: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    ids = df[split_column].unique()
    kf = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    splits = []
    for train_idx, test_idx in kf.split(ids):
        train_ids = ids[train_idx]
        test_ids = ids[test_idx]
        df_train = df[df[split_column].isin(train_ids)]
        df_test = df[df[split_column].isin(test_ids)]
        splits.append((df_train, df_test))

    return splits


def process_mask(
    df: pd.DataFrame,
    save_dir: str,
    smooth_mask: bool = True,
    save_color_mask: bool = True,
) -> None:
    if len(df) == 0:
        return

    first_row = df.iloc[0]
    img_name = os.path.basename(first_row.img_path)
    img_height, img_width = first_row.img_height, first_row.img_width

    mask = np.zeros((img_height, img_width))
    mask_color = np.zeros((img_height, img_width, 3), dtype='uint8')
    mask_color[:] = (128, 128, 128)

    mask_processor = MaskProcessor() if smooth_mask else None

    for idx, obj in enumerate(df.itertuples(index=False)):
        obj_mask = convert_base64_to_numpy(obj.encoded_mask).astype('uint8')
        if smooth_mask:
            obj_mask = mask_processor.smooth_mask(mask=obj_mask)
            obj_mask = mask_processor.remove_artifacts(mask=obj_mask)
        mask[obj_mask == 1] = CLASS_ID[obj.class_name]
        mask_color[mask == CLASS_ID[obj.class_name]] = CLASS_COLOR_BGR[obj.class_name]

    # Save the indexed mask
    cv2.imwrite(os.path.join(save_dir, 'mask', img_name), mask)

    # Save the color mask only if save_color_mask is True
    if save_color_mask:
        cv2.imwrite(os.path.join(save_dir, 'mask_color', img_name), mask_color)

    # Copy the image to the destination directory
    shutil.copy(first_row.img_path, os.path.join(save_dir, 'img', img_name))


def merge_and_save_metadata(
    dfs: List[pd.DataFrame],
    save_dir: str,
) -> None:
    # Merge data frames
    df = pd.concat(dfs).reset_index(drop=True)
    df.index += 1
    # Save metadata
    os.makedirs(save_dir, exist_ok=True)
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
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # Create directories for storing images and masks
    create_data_directories(
        num_folds=cfg.num_folds,
        save_color_mask=cfg.save_color_mask,
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
        split_column=cfg.split_column,
        num_folds=cfg.num_folds,
        seed=cfg.seed,
    )

    dfs = []
    for fold_idx, (df_train, df_test) in enumerate(splits, start=1):
        # Update metadata
        df = update_metadata(
            df_train=df_train,
            df_test=df_test,
            fold_idx=fold_idx,
        )
        dfs.append(df)

        gb_train = df_train.groupby('img_path')
        gb_test = df_test.groupby('img_path')
        train_studies_count = df.loc[df['split'] == 'train', 'study'].nunique()
        test_studies_count = df.loc[df['split'] == 'test', 'study'].nunique()
        log.info('')
        log.info(
            f'Fold {fold_idx} - Train studies / images...: {train_studies_count} / {len(gb_train)}',
        )
        log.info(
            f'Fold {fold_idx} - Test studies / images....: {test_studies_count} / {len(gb_test)}',
        )

        # Process train and test subsets
        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_color_mask=cfg.save_color_mask,
                save_dir=f'{save_dir}/fold_{fold_idx}/train',
            )
            for _, df in tqdm(gb_train, desc=f'Process train subset - Fold {fold_idx}')
        )

        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_color_mask=cfg.save_color_mask,
                save_dir=f'{save_dir}/fold_{fold_idx}/test',
            )
            for _, df in tqdm(gb_test, desc=f'Process test subset - Fold {fold_idx}')
        )

    # Merge fold dataframes and save as a single CSV file
    merge_and_save_metadata(
        dfs=dfs,
        save_dir=save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
