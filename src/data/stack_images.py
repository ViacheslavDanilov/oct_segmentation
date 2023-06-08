import logging
import os
from glob import glob
from pathlib import Path
from typing import List

import cv2
import ffmpeg
import hydra
import imutils
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_dir_list, get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_single_series(
    series_dirs: List[str],
    img_height: int,
    img_width: int,
    output_type: str,
    fps: int,
    save_dir: str,
):
    series_name = Path(series_dirs[0]).parts[-1]
    study_name = Path(series_dirs[0]).parts[-2]
    if output_type == 'video':
        save_dir_video = os.path.join(save_dir, study_name)
        os.makedirs(save_dir_video, exist_ok=True)
    else:
        save_dir_img = os.path.join(save_dir, study_name, series_name)
        os.makedirs(save_dir_img, exist_ok=True)

    img_list = []
    for img_dir in series_dirs:
        img_list_ = get_file_list(
            src_dirs=img_dir,
            ext_list='.png',
        )
        img_list.append(img_list_)

    # Create video writer
    if output_type == 'video':
        video_path_temp = os.path.join(
            save_dir_video,
            f'{study_name}_{series_name}_temp.mp4',
        )
        video = cv2.VideoWriter(
            video_path_temp,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (
                len(img_list) * img_width,
                img_height,
            ),  # To keep aspect ratio: len(img_list) * img_height
        )

    # Iterate over images
    for slice, img_paths in enumerate(zip(*img_list)):
        img_out = np.zeros([img_height, 1, 3], dtype=np.uint8)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img.shape[0] != img_height or img.shape[1] != img_width:
                img = imutils.resize(img, height=img_height, inter=cv2.INTER_LINEAR)
            img_out = np.hstack([img_out, img])
        img_out = np.delete(img_out, 0, 1)

        if output_type == 'image':
            img_name = f'{study_name}_{series_name}_{slice + 1:03d}.png'
            img_save_path = os.path.join(save_dir_img, img_name)
            cv2.imwrite(img_save_path, img_out)
        elif output_type == 'video':
            video.write(img_out)
        else:
            raise ValueError(f'Unknown output_type value: {output_type}')

    video.release() if output_type == 'video' else False

    # Replace OpenCV videos with FFmpeg ones
    if output_type == 'video':
        video_path = os.path.join(save_dir_video, f'{study_name}_{series_name}.mp4')
        stream = ffmpeg.input(video_path_temp)
        stream = ffmpeg.output(stream, video_path, vcodec='libx264', video_bitrate='10M')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        os.remove(video_path_temp)

    if output_type == 'video':
        log.info(f'Series {study_name}/{series_name} converted and saved to {video_path}')
    else:
        log.info(f'Series {study_name}/{series_name} converted and saved to {save_dir_img}')


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='stack_images',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Get list of RGB and grayscale studies
    study_list_rgb = get_dir_list(
        data_dir=cfg.data_dir_1,
        include_dirs=cfg.include_dirs,
        exclude_dirs=cfg.exclude_dirs,
    )
    series_dirs_ = [
        glob(study_list_rgb[idx] + '*/', recursive=True) for idx in range(len(study_list_rgb))
    ]
    series_dirs_rgb: List[str] = sum(series_dirs_, [])

    study_list_gray = get_dir_list(
        data_dir=cfg.data_dir_2,
        include_dirs=cfg.include_dirs,
        exclude_dirs=cfg.exclude_dirs,
    )
    series_dirs_ = [
        glob(study_list_gray[idx] + '*/', recursive=True) for idx in range(len(study_list_gray))
    ]
    series_dirs_gray: List[str] = sum(series_dirs_, [])

    # Get paired study list
    assert len(study_list_rgb) == len(study_list_gray), 'Mismatch number of studies'
    assert len(study_list_rgb) == len(study_list_gray), 'Mismatch number of series'
    series_list = [[series_dirs_rgb[i], series_dirs_gray[i]] for i in range(len(series_dirs_rgb))]

    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_single_series)(
            series_dirs=series_dirs,
            img_height=cfg.output_size[0],
            img_width=cfg.output_size[1],
            output_type=cfg.output_type,
            fps=cfg.fps,
            save_dir=cfg.save_dir,
        )
        for series_dirs in tqdm(series_list, desc='Stacking series', unit=' series')
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
