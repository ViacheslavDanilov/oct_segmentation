import json
import logging
import os
import time
from typing import List

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_IDS, data_processing, preprocessing_img, save_results
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import pick_device

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

models_meta = {
    'Lumen': {
        'model_dir': 'LM',
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


def images_segmentation(
    images: List[Image],
    masks: List[np.ndarray],
    output_size: List[int],
    classes: List[str],
    model_dir: str,
    device: str,
) -> List[np.ndarray]:
    for class_name in classes:
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
            predict_mask = cv2.resize(predict_mask, output_size)
            if len(predict_mask.shape) > 2:
                predict_mask = predict_mask[:, :, models_meta[class_name]['index']]
            mask[:, :, CLASS_IDS[class_name] - 1] = predict_mask
    return masks


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
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    model_dir = os.path.join(PROJECT_DIR, cfg.model_dir)

    # Dataset processing
    start = time.time()
    images, masks, images_name = data_processing(
        data_path=os.path.join(PROJECT_DIR, cfg.data_path),
        save_dir=save_dir,
        output_size=cfg.output_size,
    )
    log.info(f'Number of images: {len(images_name)}')

    # Perform inference on the dataset
    start_inference = time.time()
    masks = images_segmentation(
        images=images,
        masks=masks,
        output_size=cfg.output_size,
        classes=cfg.classes,
        model_dir=model_dir,
        device=device,
    )
    end_inference = time.time()

    # Result processing
    save_results(
        images=images,
        masks=masks,
        images_name=images_name,
        classes=cfg.classes,
        save_dir=save_dir,
    )

    log.info(f'Prediction time: {end_inference - start_inference:.1f} s')
    log.info(f'Overall computation time: {time.time() - start:.1f} s')
    log.info('Complete')


if __name__ == '__main__':
    main()
