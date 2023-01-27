import os
import shutil
from glob import glob

import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig):
    work_dir = cfg.meta.study_dir
    patients = os.listdir(work_dir)
    for patient_num in patients:
        if os.path.isdir(f'{work_dir}/{patient_num}'):
            d_imgs = glob(f'{work_dir}/{patient_num}/IMG*')
            if len(d_imgs) == 1:
                os.rename(f'{work_dir}/{patient_num}', f'{work_dir}/{patient_num}_01')
            else:
                for id, d_img in enumerate(d_imgs):
                    new_dir_name = f'{work_dir}/{patient_num}_{str(id + 1).zfill(2)}'
                    os.makedirs(new_dir_name)
                    shutil.copy(f'{work_dir}/{patient_num}/DICOMDIR', f'{new_dir_name}/DICOMDIR')
                    shutil.copy(d_img, f'{new_dir_name}/IMG001')
                shutil.rmtree(f'{work_dir}/{patient_num}')


if __name__ == '__main__':
    main()
