import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from typing import List, Tuple

import cv2
import imutils
import numpy as np
from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import get_dir_list, get_file_list

logger = logging.getLogger(__name__)


def process_single_study(
        study_dir: str,
        img_height: int,
        img_width: int,
        output_type: str,
        fps: int,
):

    suffix = 'stack'
    img_dirs = glob(study_dir + '*/', recursive=True)
    img_dirs = list(filter(lambda x: suffix not in x, img_dirs))
    img_dirs.sort()

    # Select save_dir based on output_type
    study_name = Path(study_dir).stem
    if output_type == 'video':
        save_dir = Path(study_dir)
    else:
        save_dir = os.path.join(Path(study_dir), 'images_{:s}'.format(suffix))
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
        video_name = '{:s}_{:s}.mp4'.format(study_name, suffix)
        video_path = os.path.join(save_dir, video_name)

        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (len(img_list) * img_width, img_height),       # To keep aspect ratio: len(img_list) * img_height
        )

    # Iterate over images
    for slice, img_paths in enumerate(zip(*img_list)):

        img_out = np.zeros([img_height, 1, 3], dtype=np.uint8)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if (
                    img.shape[0] != img_height
                    or img.shape[1] != img_width
            ):
                img = imutils.resize(img, height=img_height, inter=cv2.INTER_LINEAR)
            img_out = np.hstack([img_out, img])
        img_out = np.delete(img_out, 0, 1)

        if output_type == 'images':
            img_name = '{:s}_{:04d}.png'.format(study_name, slice + 1)
            img_save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_save_path, img_out)
        elif output_type == 'video':
            video.write(img_out)
        else:
            raise ValueError('Unknown output_type value: {:s}'.format(output_type))

    video.release() if output_type == 'video' else False
    if output_type == 'video':
        logger.info('Study {:s} converted and saved to {:s}'.format(study_name, video_path))
    else:
        logger.info('Study {:s} converted and saved to {:s}'.format(study_name, save_dir))


def main(
        study_list: List[str],
        output_size: Tuple[int, int],
        output_type: str,
        fps: int,
) -> None:
    num_cores = multiprocessing.cpu_count()
    processing_func = partial(
        process_single_study,
        img_height=output_size[0],
        img_width=output_size[1],
        output_type=output_type,
        fps=fps,
    )
    process_map(
        processing_func,
        tqdm(study_list, desc='Stacking studies', unit=' study'),
        max_workers=num_cores,
    )


if __name__ == "__main__":

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Arrange annotations')
    parser.add_argument('--study_dir', required=True, type=str, help='directory with studies')
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--output_size', nargs='+', default=[1000, 1000], type=int)
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
        fps=args.fps,
    )

    logger.info('Stacking complete')
