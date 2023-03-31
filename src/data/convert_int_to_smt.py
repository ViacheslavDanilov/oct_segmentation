import json
import multiprocessing
import os
from typing import Any, List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig
from supervisely import Polygon
from tqdm import tqdm


def get_mask(
        img_path: str,
        classes: List[str],
        data: pd.DataFrame,
        save_dir: str,
) -> None:
    if len(data) > 0 and len(list(set(classes) & set(data.Class.unique()))) > 0:
        mask = np.zeros((int(data.Image_width.mean()), int(data.Image_height.mean())))
        for _, row in data.iterrows():
            if row.Class in classes:
                figure_data = sly.Bitmap.base64_2_data(row.Mask)
                mask[figure_data == True] = row.Class_ID + 1
        cv2.imwrite(f'{save_dir}/{os.path.basename(img_path)}', mask)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='convert_int_to_smt',
    version_base=None,
)
def main(
        cfg: DictConfig,
):
    data_path = os.path.join(cfg.study_dir, cfg.df_name)
    df = pd.read_excel(data_path)
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df['Image_path'] == img_path],
            save_dir=cfg.save_dir
        )
        for img_path in tqdm(df.Image_path.unique(), desc='img analysis')
    )


if __name__ == '__main__':
    main()
