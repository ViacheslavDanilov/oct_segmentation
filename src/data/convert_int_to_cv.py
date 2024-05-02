import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import hydra
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.mask_processor import MaskProcessor
from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS, convert_base64_to_numpy

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
    classes: List[str] = None,
) -> pd.DataFrame:
    if classes is not None:
        df = df[df['class_name'].isin(classes)]

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


def colorize_mask(
    mask: np.ndarray,
    classes: List[str],
    background: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    mask_color[:] = background

    for idx, class_name in enumerate(classes):
        channel_id = CLASS_IDS[class_name] - 1  # type: ignore
        mask_color[mask[:, :, channel_id] == 255] = CLASS_COLORS_RGB[class_name]

    return mask_color


def apply_circle_crop(
    img: np.ndarray,
    crop: List[List[int]],
    background: Union[Tuple[int, ...], int] = 0,
) -> np.ndarray:
    """Crop an image with a circular region defined by the crop coordinates.

    Args:
    - img: Input image (numpy array).
    - crop: List containing the crop coordinates: [[x1, y1], [x2, y2]].
    - background: Background value to fill the cropped area with. If a tuple, it specifies the background color for each channel.

    Returns:
    - cropped_img: Cropped and masked image.
    """
    # Extract crop coordinates
    x1, y1 = crop[0]
    x2, y2 = crop[1]

    # Calculate center and radii of the ellipse
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    radius_x = abs(x2 - x1) // 2
    radius_y = abs(y2 - y1) // 2

    # Create a mask with the same dimensions as the image
    circular_mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

    # Draw a filled ellipse on the mask
    cv2.ellipse(
        circular_mask,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        360,
        (255, 255, 255),
        -1,
    )

    # Apply the mask to the image
    masked_img = np.zeros_like(img, dtype=np.uint8)
    for channel in range(img.shape[2]):
        masked_img[:, :, channel] = cv2.bitwise_and(img[:, :, channel], circular_mask)

    # Create a mask for the background
    background_mask = cv2.bitwise_not(circular_mask)

    # If the background value is a single value, repeat it for each channel
    if isinstance(background, int):
        background = (background,) * img.shape[2]

    # Fill the background with the specified value
    for channel in range(img.shape[2]):
        masked_img[:, :, channel] += background[channel] * background_mask  # type: ignore

    # Crop the image to the exact crop coordinates
    cropped_img = masked_img[y1:y2, x1:x2]

    return cropped_img


def process_pair(
    df: pd.DataFrame,
    save_dir: str,
    crop: List[List[int]],
    classes: List[str],
    smooth_mask: bool = True,
    save_color_mask: bool = True,
) -> None:
    if len(df) == 0:
        return

    img_path = df.iloc[0].img_path
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    num_classes = len(classes)
    mask = np.zeros((img_height, img_width, num_classes), dtype='uint8')

    mask_processor = MaskProcessor() if smooth_mask else None

    for idx, obj in enumerate(df.itertuples(index=False)):
        obj_mask = convert_base64_to_numpy(obj.encoded_mask).astype('uint8')
        if smooth_mask:
            obj_mask = mask_processor.smooth_mask(mask=obj_mask)
            obj_mask = mask_processor.remove_artifacts(mask=obj_mask)
        channel_id = CLASS_IDS[obj.class_name] - 1  # type: ignore
        mask[:, :, channel_id][obj_mask == 1] = 255

    # Colorize mask
    mask_color = colorize_mask(mask=mask, classes=classes)

    # Apply the circular mask to the image and masks
    if crop is not None:
        img = apply_circle_crop(img, crop, background=0)
        mask = apply_circle_crop(mask, crop, background=0)
        mask_color = apply_circle_crop(mask_color, crop, background=128)

    # Save image and masks
    basename = Path(img_path).stem
    cv2.imwrite(os.path.join(save_dir, 'img', f'{basename}.png'), img)
    tifffile.imwrite(os.path.join(save_dir, 'mask', f'{basename}.tiff'), mask, compression='LZW')
    if save_color_mask:
        tifffile.imwrite(
            os.path.join(save_dir, 'mask_color', f'{basename}.tiff'),
            mask_color,
            compression='LZW',
        )


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
        classes=cfg.classes,
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
            delayed(process_pair)(
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_color_mask=cfg.save_color_mask,
                crop=cfg.crop,
                classes=cfg.classes,
                save_dir=f'{save_dir}/fold_{fold_idx}/train',
            )
            for _, df in tqdm(gb_train, desc=f'Process train subset - Fold {fold_idx}')
        )

        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_pair)(
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_color_mask=cfg.save_color_mask,
                crop=cfg.crop,
                classes=cfg.classes,
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
