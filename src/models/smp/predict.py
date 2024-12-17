import json
import logging
import os
import time
from glob import glob
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import get_img_mask_union_pil

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
models_meta = {
    'Lumen': {
        'model_dir': 'LN',
        'index': 0,
    },
    'Lipid core': {
        'model_dir': 'FC_LC',
        'index': 0,
    },
    'Fibrous cap': {
        'model_dir': 'FC_LC',
        'index': 1,
    },
    'Vasa vasorum': {
        'model_dir': 'VV',
        'index': 0,
    },
}


def preprocessing_img(
    img: Image,
    input_size: int,
):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (input_size, input_size))
    return image


def pick_device(
    option: str,
) -> str:
    """Pick the appropriate device based on the provided option.

    Args:
        option (str): Available device option ('cpu', 'cuda', 'auto').

    Returns:
        str: Selected device.
    """
    if option == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif option in ['cpu', 'cuda']:
        return option
    else:
        raise ValueError("Invalid device option. Please specify 'cpu', 'cuda', or 'auto'.")


def process_mask(
    source_image: Image,
    output_size: List[int],
    classes: List[str],
    classes_model: List[str],
    mask: np.ndarray,
) -> Tuple[Image, Image]:
    """Process the predicted mask to overlay on the source image and generate a color mask.

    Args:
        source_image: The original histology image.
        output_size: The desired output size for the image and mask.
        classes: The list of classes to be considered for segmentation.
        classes_model: The list of classes that the model can predict.
        mask: The predicted mask from the model.

    Returns:
        A tuple containing the overlay image and the color mask image.
    """
    # Resize the source image to the output size
    source_image = source_image.resize((output_size[0], output_size[1]))
    # Initialize a color mask with a default color
    color_mask = np.zeros((source_image.size[0], source_image.size[1], 3))
    mask = cv2.resize(mask, (output_size[0], output_size[1]))
    color_mask[:, :] = (128, 128, 128)  # Default color

    # Overlay the mask on the source image for the specified classes
    for idx, cl in enumerate(classes_model):
        if cl in classes:
            source_image = get_img_mask_union_pil(
                img=source_image,
                mask=mask[:, :, idx].copy(),
                alpha=0.85,
                color=CLASS_COLOR[cl],  # type: ignore
            )
            color_mask[mask[:, :, idx] == 1] = CLASS_COLOR[cl]
    return source_image, Image.fromarray(color_mask.astype('uint8'))


# def data_generator(
#     data: List[str],
#     batch_size: int,
# ) -> Generator[List[str], None, None]:
#     """Generator function to yield batches of data paths.
#
#     Args:
#         data: A list of data paths.
#         batch_size: The size of each batch.
#
#     Yields:
#         A batch of data paths.
#     """
#     while data:
#         batch = []
#         for _ in range(batch_size):
#             if data:
#                 img_path = data.pop()
#                 batch.append(img_path)
#             else:
#                 break
#         yield batch


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main function to perform histology image segmentation prediction.

    Args:
        cfg: Configuration parameters loaded from a YAML file using Hydra.
    """
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Pick the appropriate device based on the provided option
    device = pick_device(option=cfg.device)

    # Define absolute paths for data, save directory, and model directory
    data_path = os.path.join(PROJECT_DIR, cfg.data_path)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    model_dir = os.path.join(PROJECT_DIR, cfg.model_dir)

    # Dataset processing
    start = time.time()
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(data_path):
        images_path = [data_path]
    else:
        images_path = glob(f'{data_path}/*.[pj][np][ge]*')
    log.info(f'Number of images: {len(images_path)}')

    images, masks, images_name = [], [], []
    for img_path in tqdm(
        images_path,
        total=len(images_path),
        desc='Image processing',
        unit='image',
    ):
        img = Image.open(img_path).resize(cfg.output_size)
        mask = np.zeros((cfg.output_size[0], cfg.output_size[1], 4))
        images.append(img)
        masks.append(mask)
        images_name.append(os.path.basename(img_path).split('.')[0])
    start_inference = time.time()

    # Perform inference on the dataset
    for class_name in cfg.classes:
        # Load model configuration and initialize the model
        model_weights = f"{model_dir}/{models_meta[class_name]['model_dir']}/weights.ckpt"
        with open(f"{model_dir}/{models_meta[class_name]['model_dir']}/config.json", 'r') as file:
            model_cfg = json.load(file)
        start_load = time.time()
        model = OCTSegmentationModel.load_from_checkpoint(
            checkpoint_path=model_weights,
            encoder_weights=None,
            arch=model_cfg['architecture'],
            encoder_name=model_cfg['encoder'],
            model_name=model_cfg['model_name'],
            in_channels=3,
            classes=model_cfg['classes'],
            map_location='cuda:0' if device == 'cuda' else device,
        )
        model.eval()
        log.info(
            f"{model_cfg['model_name']} loaded successfully. Time taken: {time.time() - start_load:.1f} s",
        )
        for img, mask in tqdm(
            zip(images, masks),
            total=len(images),
            desc=f'Image segmentation {class_name}',
            unit='image',
        ):
            image = preprocessing_img(
                img=img.copy(),
                input_size=model_cfg['input_size'],
            )
            predict_mask = model.predict(
                images=np.array([image]),
                device='cuda',
            )[0]
            predict_mask = cv2.resize(predict_mask, cfg.output_size)
            if len(predict_mask.shape) > 2:
                predict_mask = predict_mask[:, :, models_meta[class_name]['index']]
            mask[:, :, CLASS_IDS[class_name] - 1] = predict_mask
    end_inference = time.time()

    # Result processing
    for img, mask, image_name in tqdm(
        zip(images, masks, images_name),
        total=len(images),
        desc='Image|Mask postprocessing',
        unit='image',
    ):
        color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
        for class_name in cfg.classes:
            m = mask[:, :, CLASS_IDS[class_name] - 1]
            m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
            m = cv2.GaussianBlur(m, (5, 5), 0)
            img = get_img_mask_union_pil(
                img=img,
                mask=m * 64,
                color=CLASS_COLORS_RGB[class_name],
            )
            m = mask[:, :, CLASS_IDS[class_name] - 1] * 255
            class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[class_name])
            color_mask.paste(class_img, (0, 0), Image.fromarray(m).convert('L'))
        color_mask.save(f'{save_dir}/{image_name}_mask.png')
        img.save(f'{save_dir}/{image_name}_overlay.png')

    log.info(f'Prediction time: {end_inference - start_inference:.1f} s')
    log.info(f'Overall computation time: {time.time() - start:.1f} s')
    log.info('Complete')


if __name__ == '__main__':
    main()
