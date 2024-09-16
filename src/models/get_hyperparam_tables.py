import os
from typing import List

import pandas as pd


def get_best_architectures(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Group by Class and Architecture, then find the row with max DSC
    best_architectures = df.loc[df.groupby(['Class', 'Architecture'])['DSC'].idxmax()]

    # Reset index for better readability
    best_architectures.reset_index(drop=True, inplace=True)

    return best_architectures


def combine_excel_files(
    dir_path: str,
) -> pd.DataFrame:
    # List to store all dataframes
    all_dataframes: List[pd.DataFrame] = []

    # Get all xlsx files in the directory
    excel_files = [f for f in os.listdir(dir_path) if f.endswith('.xlsx')]

    # Read each Excel file
    for file in excel_files:
        df = pd.read_excel(os.path.join(dir_path, file))
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


# Usage
dir_path = 'eval/tuning'
output_file_1 = 'eval/tuning/hyperparameters_all.xlsx'
output_file_2 = 'eval/tuning/hyperparameters_best.xlsx'

os.makedirs(dir_path, exist_ok=True)

combined_df = combine_excel_files(dir_path)
combined_df.to_excel(output_file_1, index=False)

best_configs = get_best_architectures(combined_df)
best_configs.to_excel(output_file_2, index=False)
