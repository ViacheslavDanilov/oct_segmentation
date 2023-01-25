import os
import hydra
from glob import glob

from omegaconf import DictConfig


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data_sly', version_base=None)
def main(
    cfg: DictConfig,
):
    project_name = os.path.basename(cfg.meta.study_dir)
    for dataset_fs in os.listdir(cfg.meta.study_dir):
        if os.path.isdir(f'{cfg.meta.study_dir}/{dataset_fs}'):
            names, video_paths, ann_paths = [], [], []
            for item_name in glob():
                img_path, ann_path = dataset_fs.get_item_paths(item_name)
                names.append(item_name)
                img_paths.append(img_path)
                ann_paths.append(ann_path)


if __name__ == '__main__':
    main()