import base64
import logging
import os
import zlib
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.smp.utils import get_img_mask_union_pil

CLASS_MAP = {
    'Lumen': {
        'id': 1,
        'color': [228, 30, 199],
    },
    'Fibrous cap': {
        'id': 2,
        'color': [123, 171, 226],
    },
    'Lipid core': {
        'id': 3,
        'color': [125, 227, 127],
    },
    'Vasa vasorum': {
        'id': 4,
        'color': [208, 2, 27],
    },
}

CLASS_COLORS_RGB = {
    class_name: tuple(class_info['color']) for class_name, class_info in CLASS_MAP.items()  # type: ignore
}

CLASS_COLORS_BGR = {
    class_name: tuple(class_info['color'][::-1]) for class_name, class_info in CLASS_MAP.items()  # type: ignore
}

CLASS_IDS = {class_name: class_info['id'] for class_name, class_info in CLASS_MAP.items()}

CLASS_IDS_REVERSED = dict((v, k) for k, v in CLASS_IDS.items())


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    filename_template: str = '',
) -> List[str]:
    """Get a list of files in the specified directory with specific extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
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
                if file_ext in ext_list and filename_template in file:
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


def convert_base64_to_numpy(
    s: str,
) -> np.ndarray:
    """Convert base64 encoded string to numpy array.

    import supervisely as sly
    encoded_string = 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
    figure_data = sly.Bitmap.base64_2_data(encoded_string)
    print(figure_data)
    #  [[ True  True  True]
    #   [ True False  True]
    #   [ True  True  True]]
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(bool)  # 4-channel images
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(bool)  # flat 2D mask
    else:
        raise RuntimeError('Wrong internal mask format')
    return mask


def preprocessing_img(
    img: Image,
    input_size: int,
):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (input_size, input_size))
    return image


def data_processing(
    data_path: str,
    save_dir: str,
    output_size: List[int],
) -> Tuple[List[Image], List[np.ndarray], List[str]]:
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(data_path):
        images_path = [data_path]
    else:
        images_path = glob(f'{data_path}/*.[pj][np][ge]*')

    images, masks, image_names = [], [], []
    for img_path in tqdm(
        images_path,
        total=len(images_path),
        desc='Image processing',
        unit='image',
    ):
        img = Image.open(img_path).resize(output_size)
        mask = np.zeros((output_size[0], output_size[1], 4))
        images.append(img)
        masks.append(mask)
        image_names.append(os.path.basename(img_path).split('.')[0])
    return images, masks, image_names


def save_results(
    images: List[Image],
    masks: List[np.ndarray],
    images_name: List[str],
    classes: List[str],
    save_dir: str,
) -> None:
    for img, mask, image_name in tqdm(
        zip(images, masks, images_name),
        total=len(images),
        desc='Image & mask post-processing',
        unit='image',
    ):
        color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
        for class_name in classes:
            m = mask[:, :, CLASS_IDS[class_name] - 1]  # type: ignore
            m = cv2.morphologyEx(
                m,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                3,
            )
            m_d = cv2.dilate(m.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            m_e = cv2.erode(m.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            m = cv2.GaussianBlur(m, (5, 5), 0)
            m_d[m_e > 0] = 0
            img = get_img_mask_union_pil(
                img=img,
                mask=m * 64,
                color=CLASS_COLORS_RGB[class_name],
            )
            img = get_img_mask_union_pil(
                img=img,
                mask=m_d * 255,
                color=CLASS_COLORS_RGB[class_name],
            )
            m = mask[:, :, CLASS_IDS[class_name] - 1] * 255  # type: ignore
            class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[class_name])
            color_mask.paste(class_img, (0, 0), Image.fromarray(m).convert('L'))
        color_mask.save(f'{save_dir}/{image_name}_mask.png')
        img.save(f'{save_dir}/{image_name}_overlay.png')
