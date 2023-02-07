import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    dirname_template: str = '',
    filename_template: str = '',
) -> List[str]:
    """Get a list of files in the specified directory with specific extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        dirname_template: include directories with this template
        filename_template: include files with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                dir_name = os.path.basename(root)
                if (
                    file_ext in ext_list
                    and dirname_template in dir_name
                    and filename_template in file
                ):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


def get_dir_list(
    data_dir: str,
    include_dirs: List[str],
    exclude_dirs: List[str],
) -> List[str]:
    dir_list = []
    _dir_list = glob(data_dir + '/*/')
    for series_dir in _dir_list:
        if include_dirs and Path(series_dir).name not in include_dirs:
            logging.info(
                f'Skip {Path(series_dir).name} because it is not in the included_dirs list',
            )
            continue

        if exclude_dirs and Path(series_dir).name in exclude_dirs:
            logging.info(
                f'Skip {Path(series_dir).name} because it is in the excluded_dirs list',
            )
            continue

        dir_list.append(series_dir)
    dir_list.sort()
    return dir_list


def convert_to_grayscale(
    img_src: np.ndarray,
    min_limit: int = 40,
    max_limit: int = 220,
) -> np.ndarray:

    img = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    img[img < min_limit] = 0
    img[img > max_limit] = 255
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def get_study_name(
    dcm_path: str,
) -> str:
    study_name = Path(dcm_path).parts[-2]
    return study_name


def get_series_name(
    dcm_path: str,
) -> str:
    dcm_name = Path(dcm_path).name
    series_name_ = dcm_name.replace('IMG', '')
    series_name = str(int(series_name_))
    return series_name
