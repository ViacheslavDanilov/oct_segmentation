import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Tuple, List

import cv2
import pydicom
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import get_dir_list, convert_to_grayscale

logger = logging.getLogger(__name__)


def convert_single_study(
        study_dir: str,
        output_type: str,
        output_size: Tuple[int, int],
        to_gray: bool,
        fps: int,
) -> None:
    dcm_path = os.path.join(study_dir, 'IMG001')            # FIXME: temporal solution
    study = pydicom.dcmread(dcm_path)
    dcm = study.pixel_array
    slices = dcm.shape[0]

    suffix = '_gray' if to_gray else ''

    # Select save_dir based on output_type
    study_name = Path(study_dir).stem
    if output_type == 'video':
        save_dir = Path(study_dir)
    else:
        save_dir = os.path.join(Path(study_dir), 'images{:s}'.format(suffix))
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False

    # Create video writer
    if output_type == 'video':
        video_height, video_width = output_size
        video_name = '{:s}{:s}.mp4'.format(study_name, suffix)
        video_path = os.path.join(save_dir, video_name)
        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (video_width, video_height),
        )

    # Iterate over DICOM slices
    for slice in range(slices):
        img = dcm[slice, :, :, :]
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = convert_to_grayscale(img, min_limit=40, max_limit=220) if to_gray else img

        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != output_size:
            img = cv2.resize(img, output_size)

        if output_type == 'images':
            img_name = '{:s}_{:04d}.png'.format(study_name, slice+1)
            img_save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_save_path, img)
        elif output_type == 'video':
            video.write(img)
        else:
            raise ValueError('Unknown output_type value: {:s}'.format(output_type))

    video.release() if output_type == 'video' else False
    if output_type == 'video':
        logger.info('Study {:s} converted and saved to {:s}'.format(study_name, video_path))
    else:
        logger.info('Study {:s} converted and saved to {:s}'.format(study_name, save_dir))


def main(
        study_list: List[str],
        output_type: str,
        output_size: Tuple[int, int],
        to_gray: bool,
        fps: int,
) -> None:
    num_cores = multiprocessing.cpu_count()
    conversion_func = partial(
        convert_single_study,
        output_type=output_type,
        output_size=output_size,
        to_gray=to_gray,
        fps=fps,
    )
    process_map(
        conversion_func,
        tqdm(study_list, desc='Convert studies', unit=' study'),
        max_workers=num_cores,
    )


if __name__ == '__main__':

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Convert DICOM files to images or video')
    parser.add_argument('--study_dir', required=True, type=str, help='directory with studies')
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--output_size', nargs='+', default=[1000, 1000], type=int)
    parser.add_argument('--to_gray', action='store_true')
    parser.add_argument('--output_type', default='images', type=str, help='images or video')
    parser.add_argument('--fps', default=15, type=int)
    args = parser.parse_args()

    # Include or exclude specific directories
    study_list = get_dir_list(
        data_dir=args.study_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        study_list=study_list,
        output_size=tuple(args.output_size),
        output_type=args.output_type,
        to_gray=args.to_gray,
        fps=args.fps,
    )

    logger.info('DICOM conversion complete')
