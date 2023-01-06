import logging
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import ffmpeg
import hydra
import imutils
import pydicom
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.data.utils import convert_to_grayscale, get_dir_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def convert_single_study(
    study_dir: str,
    output_type: str,
    output_size: Tuple[int, int],
    to_gray: bool,
    fps: int,
    save_dir: str,
) -> None:
    dcm_path = os.path.join(study_dir, 'IMG001')  # FIXME: temporal solution
    study = pydicom.dcmread(dcm_path)
    dcm = study.pixel_array
    slices = dcm.shape[0]

    suffix = '_gray' if to_gray else ''

    # Select save_dir based on output_type
    study_name = Path(study_dir).stem
    if output_type == 'video':
        save_dir = os.path.join(save_dir, study_name)
    else:
        save_dir = os.path.join(save_dir, study_name, f'images{suffix}')
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False

    # Create video writer
    if output_type == 'video':
        video_path_temp = os.path.join(save_dir, f'{study_name}{suffix}_temp.mp4')
        video_height, video_width = output_size
        video = cv2.VideoWriter(
            video_path_temp,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (video_width, video_height),
        )

    # Iterate over DICOM slices
    for slice in range(slices):
        img = dcm[slice, :, :, :]
        img = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = convert_to_grayscale(img, min_limit=40, max_limit=220) if to_gray else img

        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != output_size:
            img = imutils.resize(img, height=output_size[0], inter=cv2.INTER_LINEAR)

        if output_type == 'image':
            img_name = f'{study_name}_{slice+1:04d}.png'
            img_save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_save_path, img)
        elif output_type == 'video':
            video.write(img)
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
        data_dir=cfg.conversion.study_dir,
        include_dirs=cfg.conversion.include_dirs,
        exclude_dirs=cfg.conversion.exclude_dirs,
    )

    num_cores = multiprocessing.cpu_count()
    conversion_func = partial(
        convert_single_study,
        output_type=cfg.conversion.output_type,
        output_size=cfg.conversion.output_size,
        to_gray=cfg.conversion.to_gray,
        fps=cfg.conversion.fps,
        save_dir=cfg.conversion.save_dir,
    )
    process_map(
        conversion_func,
        tqdm(study_list, desc='Convert studies', unit=' study'),
        max_workers=num_cores,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()
