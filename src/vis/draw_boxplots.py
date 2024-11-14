import logging
import os
import re
from typing import List

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.utils import get_file_list

log = logging.getLogger(__name__)


def get_best_epoch(
    df: pd.DataFrame,
    metric='Dice',
):
    # Find the index of the row with the maximum metric value for each fold
    best_rows = []
    for fold in df['Fold'].unique():
        for cls in df['Class'].unique():
            # Filter for the current fold and class
            subset = df[(df['Fold'] == fold) & (df['Class'] == cls)]
            # Check if subset is not empty, then get the row with the best metric
            if not subset.empty:
                best_row = subset.loc[subset[metric].idxmax()]
                best_rows.append(best_row)
    df_best = pd.DataFrame(best_rows)
    return df_best


def get_fold_id(path):
    match = re.search(r'fold_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return None


def read_model_metrics(
    csv_paths: List[str],
) -> pd.DataFrame:
    dfs = []
    for csv_path in csv_paths:
        df_ = pd.read_csv(csv_path)
        df_['Fold'] = get_fold_id(csv_path)
        dfs.append(df_)
    df = pd.concat(dfs)
    return df


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='draw_boxplots',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    class_dirs = [str(os.path.join(PROJECT_DIR, d)) for d in cfg.class_dirs]
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Get metric CSVs and read

    # Read DataFrame with metrics
    csv_paths = get_file_list(
        src_dirs=class_dirs,
        ext_list='.csv',
        filename_template='metrics',
    )
    df = read_model_metrics(csv_paths)

    # Drop unnecessary rows and find best epoch
    df = df[(df['Class'] != 'Mean') & (df['Split'] == cfg.split)]
    df_best = get_best_epoch(df, metric=cfg.metric)

    # Save best epochs
    os.makedirs(save_dir, exist_ok=True)
    df_best.reset_index(drop=True, inplace=True)
    save_path = os.path.join(save_dir, 'best_metrics.csv')
    df_best.to_csv(save_path, index=False)

    # Define the order of x-axis categories
    class_order = [
        'Lumen',
        'Fibrous cap',
        'Lipid core',
        'Vasa vasorum',
    ]

    # Plotting
    sns.set(style='whitegrid')

    # Plot the specified metric for each class
    plt.figure(figsize=(12, 12))

    # Define a color palette
    palette = sns.color_palette('muted')

    ax = sns.boxplot(
        x='Class',
        y=cfg.metric,
        data=df_best,
        palette=palette,
        hue='Class',
        legend=False,
        showfliers=False,
        order=class_order,
        linewidth=2.0,
    )

    # Label the y-axis with the metric name
    plt.ylabel(cfg.metric, fontsize=36)
    plt.xticks(rotation=90, fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlabel('')
    ax.set_ylim(0.4, 1)

    sns.despine()
    plt.tight_layout()

    # Save the plot as a high-quality image file in PNG format
    save_path = os.path.join(save_dir, f'Boxplot_{cfg.metric}_{cfg.split}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
