import logging
import os
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR
from src.data.utils import get_file_list

log = logging.getLogger(__name__)


def merge_metric_dataframes(
    csv_paths: List[str],
) -> pd.DataFrame:
    dfs = [pd.read_csv(csv_path) for csv_path in csv_paths]
    df_merge = pd.concat(dfs, ignore_index=True)

    return df_merge


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='draw_loss_plot',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    class_dir = str(os.path.join(PROJECT_DIR, cfg.class_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Merge metric dataframes
    csv_paths = get_file_list(
        src_dirs=class_dir,
        ext_list='.csv',
        filename_template='metrics',
    )
    df_metrics = merge_metric_dataframes(csv_paths=csv_paths)

    # Get class dataframe
    class_name = os.path.basename(class_dir)
    df_filt = df_metrics[df_metrics['Class'] == class_name]

    # Plot
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 10))

    # Customize color palette
    palette = sns.color_palette('bright', 2)

    # Draw line plots with confidence intervals
    train_metric = 'DSC' if cfg.train_metric == 'Dice' else cfg.train_metric
    test_metric = 'DSC' if cfg.test_metric == 'Dice' else cfg.test_metric
    sns.lineplot(
        data=df_filt[df_filt['Split'] == 'train'],
        x='Epoch',
        y=cfg.train_metric,
        color=palette[0],
        linewidth=3.0,
        label=f'{train_metric} (Train)',
        err_style='band',
        errorbar=('ci', 95),
    )
    sns.lineplot(
        data=df_filt[df_filt['Split'] == 'test'],
        x='Epoch',
        y=cfg.test_metric,
        color=palette[1],
        linewidth=3.0,
        label=f'{test_metric} (Test)',
        err_style='band',
        errorbar=('ci', 95),
    )

    plt.xlabel('Epoch', fontsize=36)
    plt.ylabel('Metric Value', fontsize=36)
    plt.xticks(np.arange(0, 176, 25), fontsize=30)
    plt.yticks(np.arange(0, 1.2, 0.2), fontsize=30)
    plt.legend(fontsize=26, loc='upper right')
    plt.grid(True)

    # Set coordinate axis limits
    plt.ylim(0, 1)
    plt.xlim(0, 125)
    plt.tight_layout(pad=0.9)

    # Save plot
    save_path = os.path.join(save_dir, f'{class_name}_{cfg.train_metric}_{cfg.test_metric}.png')
    plt.savefig(save_path, dpi=600)
    plt.show()
    log.info(f'{class_name} plot saved')

    log.info('Complete')


if __name__ == '__main__':
    main()
