import logging
import os
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_best_architectures(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Group by Class and Architecture, then find the row with max DSC
    best_architectures = df.loc[df.groupby(['Class', 'Architecture'])['DSC'].idxmax()]

    # Reset index for better readability
    best_architectures.reset_index(drop=True, inplace=True)

    return best_architectures


def combine_excel_files(
    tuning_file_paths: List[str],
) -> pd.DataFrame:
    # List to store all dataframes
    all_dataframes: List[pd.DataFrame] = []

    # Read each Excel file
    for file_path in tuning_file_paths:
        df = pd.read_excel(file_path)
        all_dataframes.append(df)

    # Find common columns
    common_columns = set(all_dataframes[0].columns)
    for df in all_dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))

    # Keep only common columns in each dataframe
    for i in range(len(all_dataframes)):
        all_dataframes[i] = all_dataframes[i][list(common_columns)]

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove columns
    combined_df = combined_df.drop(
        columns=[
            'train/f1',
            'train/precision',
            'train/recall',
            'train/iou',
            'train/dice',
            'train/loss',
            'test/f1',
            'test/precision',
            'test/recall',
            'test/iou',
            'test/dice',
            'test/loss',
            'best_precision_epoch',
            'best_recall_epoch',
            'best_iou_epoch',
        ],
    )

    # Rename columns
    combined_df = combined_df.rename(
        columns={
            'Unnamed: 0': 'ID',
            'classes': 'Class',
            'architecture': 'Architecture',
            'encoder': 'Encoder',
            'input_size': 'Input size',
            'optimizer': 'Optimizer',
            'lr': 'LR',
            'best_dice': 'DSC',
            'best_iou': 'IoU',
            'best_precision': 'Precision',
            'best_recall': 'Recall',
            'best_dice_epoch': 'Epoch',
        },
    )

    # Reorder columns providing a list of column names
    column_order = [
        'ID',
        'Name',
        'State',
        'Runtime',
        'Class',
        'Architecture',
        'Encoder',
        'Input size',
        'Optimizer',
        'LR',
        'DSC',
        'IoU',
        'Precision',
        'Recall',
        'Epoch',
    ]
    combined_df = combined_df[column_order]

    # Increment the id column
    combined_df['ID'] = combined_df['ID'].apply(lambda x: x + 1)

    return combined_df


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='get_configs',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    tuning_file_paths = [str(os.path.join(PROJECT_DIR, f)) for f in cfg.tuning_file_paths]
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    os.makedirs(save_dir, exist_ok=True)

    # Create a dataframe with all configs
    save_path_all = os.path.join(save_dir, 'configs_all.xlsx')
    combined_df = combine_excel_files(tuning_file_paths)
    combined_df.to_excel(save_path_all, index=False)

    # Create a dataframe with the best configs
    save_path_best = os.path.join(save_dir, 'configs_best.xlsx')
    best_configs = get_best_architectures(combined_df)
    best_configs.to_excel(save_path_best, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
