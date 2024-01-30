import logging
import os
from typing import Tuple

import cv2
import ffmpeg
import hydra
import imutils
import pydicom
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import (
    convert_to_grayscale,
    get_dir_list,
    get_file_list,
    get_series_name,
    get_study_name,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def convert_single_study(
    data_dir: str,
    output_type: str,
    output_size: Tuple[int, int],
    to_gray: bool,
    fps: int,
    save_dir: str,
) -> None:
    dcm_list = get_file_list(
        src_dirs=data_dir,
        ext_list='',
        filename_template='IMG',
    )

    for dcm_path in dcm_list:
        study = pydicom.dcmread(dcm_path)
        dcm = study.pixel_array
        slices = dcm.shape[0]

        # Select save_dir based on output_type
        study_name = get_study_name(dcm_path)
        series_name = get_series_name(dcm_path)
        if output_type == 'video':
            save_dir_video = os.path.join(save_dir, study_name)
            os.makedirs(save_dir_video, exist_ok=True)
        else:
            save_dir_img = os.path.join(save_dir, study_name, series_name)
            os.makedirs(save_dir_img, exist_ok=True)

        # Create video writer
        if output_type == 'video':
            video_path_temp = os.path.join(
                save_dir_video,
                f'{study_name}_{series_name}_temp.mp4',
            )
            video_height, video_width = output_size
            video = cv2.VideoWriter(
                video_path_temp,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (video_width, video_height),
            )

        # Iterate over DICOM slices
        for slice in range(slices):
            img = dcm[slice]
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
                img_name = f'{study_name}_{series_name}_{slice+1:03d}.png'
                img_save_path = os.path.join(save_dir_img, img_name)
                cv2.imwrite(img_save_path, img)
            elif output_type == 'video':
                video.write(img)
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
            log.info(f'DICOM {dcm_path} converted and saved to {video_path}')
        else:
            log.info(f'DICOM {dcm_path} converted and saved to {save_dir_img}')


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_dicoms',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    study_list = get_dir_list(
        data_dir=cfg.data_dir,
        include_dirs=cfg.include_dirs,
        exclude_dirs=cfg.exclude_dirs,
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(convert_single_study)(
            data_dir=study_dir,
            output_type=cfg.output_type,
            output_size=cfg.output_size,
            to_gray=cfg.to_gray,
            fps=cfg.fps,
            save_dir=cfg.save_dir,
        )
        for study_dir in tqdm(study_list, desc='Convert studies', unit=' study')
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
