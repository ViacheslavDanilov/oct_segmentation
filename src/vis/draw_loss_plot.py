import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='draw_loss_plot',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    metrics_path = str(os.path.join(PROJECT_DIR, cfg.metrics_path))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Read DataFrame with metrics
    df = pd.read_csv(metrics_path)
    gb = df.groupby('Model')

    for model_name, df_model in tqdm(gb, desc='Draw plots', unit=' model'):
        # Filter for the 'Mean' class
        df_filt = df_model[df_model['Class'] == 'Mean']

        # Plot
        sns.set(style='whitegrid')
        plt.figure(figsize=(12, 10))

        # Customize color palette
        palette = sns.color_palette('bright', 2)

        # Draw line plots with confidence intervals
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'train'],
            x='Epoch',
            y=cfg.train_metric,
            color=palette[0],
            linewidth=3.0,
            label=f'{cfg.train_metric} (Train)',
            err_style='band',
            errorbar=('ci', 95),
        )
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'test'],
            x='Epoch',
            y=cfg.test_metric,
            color=palette[1],
            linewidth=3.0,
            label=f'{cfg.test_metric} (Test)',
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
        plt.xlim(0, 175)
        plt.tight_layout(pad=0.9)

        # Save plot
        save_path = os.path.join(save_dir, f'{model_name}_loss.png')
        plt.savefig(save_path, dpi=600)
        plt.show()
        log.info(f'{model_name} plot saved')

    log.info('Complete')


if __name__ == '__main__':
    main()
