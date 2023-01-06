import logging
import multiprocessing
import os
from functools import partial
from glob import glob
from pathlib import Path

import cv2
import ffmpeg
import hydra
import imutils
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.data.utils import get_dir_list, get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_single_study(
    study_dir: str,
    img_height: int,
    img_width: int,
    output_type: str,
    fps: int,
    save_dir: str,
):

    suffix = '_stack'
    img_dirs = glob(study_dir + '*/', recursive=True)
    img_dirs = list(filter(lambda x: suffix not in x, img_dirs))
    img_dirs.sort()

    # Select save_dir based on output_type
    study_name = Path(study_dir).stem
    if output_type == 'video':
        save_dir = os.path.join(save_dir, study_name)
    else:
        save_dir = os.path.join(save_dir, study_name, f'images{suffix}')
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False

    img_list = []
    for img_dir in img_dirs:
        _img_list = get_file_list(
            src_dirs=img_dir,
            ext_list='.png',
        )
        img_list.append(_img_list)

    # Create video writer
    if output_type == 'video':
        video_path_temp = os.path.join(save_dir, f'{study_name}{suffix}_temp.mp4')
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
            img_name = f'{study_name}_{slice + 1:04d}.png'
            img_save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_save_path, img_out)
        elif output_type == 'video':
            video.write(img_out)
        else:
            raise ValueError(f'Unknown output_type value: {output_type}')

    video.release() if output_type == 'video' else False

    # Replace OpenCV videos with FFmpeg ones
    if output_type == 'video':
        video_path = os.path.join(save_dir, f'{study_name}{suffix}.mp4')
        stream = ffmpeg.input(video_path_temp)
        stream = ffmpeg.output(stream, video_path, vcodec='libx264', video_bitrate='10M')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        os.remove(video_path_temp)

    if output_type == 'video':
        log.info(f'Study {study_name} converted and saved to {video_path}')
    else:
        log.info(f'Study {study_name} converted and saved to {save_dir}')


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    study_list = get_dir_list(
        data_dir=cfg.stack.study_dir,
        include_dirs=cfg.stack.include_dirs,
        exclude_dirs=cfg.stack.exclude_dirs,
    )

    num_cores = multiprocessing.cpu_count()
    processing_func = partial(
        process_single_study,
        img_height=cfg.stack.output_size[0],
        img_width=cfg.stack.output_size[1],
        output_type=cfg.stack.output_type,
        fps=cfg.stack.fps,
        save_dir=cfg.stack.save_dir,
    )
    process_map(
        processing_func,
        tqdm(study_list, desc='Stacking studies', unit=' study'),
        max_workers=num_cores,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()