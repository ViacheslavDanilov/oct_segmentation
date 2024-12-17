import json
import logging
import os
import time
from typing import Dict, List, Tuple

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

# Model metadata
MODELS_META = {
    'Lumen': {'model_dir': 'LM', 'index': 0},
    'Lipid core': {'model_dir': 'FC_LC', 'index': 0},
    'Fibrous cap': {'model_dir': 'FC_LC', 'index': 1},
    'Vasa vasorum': {'model_dir': 'VV', 'index': 0},
}


def load_model(
    model_dir: str,
    device: str,
) -> Tuple[OCTSegmentationModel, Dict]:
    """Load a segmentation model from checkpoint."""
    with open(f'{model_dir}/config.json', 'r') as file:
        model_cfg = json.load(file)
    model_weights = f'{model_dir}/weights.ckpt'
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
    return model, model_cfg


def preprocess_images(
    images: List[Image],
    input_size: int,
) -> np.ndarray:
    """Preprocess images to match model input size."""
    return np.array([preprocessing_img(img=img.copy(), input_size=input_size) for img in images])


def segment(
    images: List[Image],
    masks: List[np.ndarray],
    output_size: List[int],
    classes: List[str],
    models_dir: str,
    device: str,
) -> List[np.ndarray]:
    """Perform segmentation for given images using specified models."""
    for class_name in classes:
        class_meta = MODELS_META[class_name]
        model_dir = os.path.join(models_dir, str(class_meta['model_dir']))

        # Load model once per class
        start_load = time.time()
        model, model_cfg = load_model(model_dir=model_dir, device=device)
        log.info(
            f"{model_cfg['architecture']} loaded successfully. Time taken: {time.time() - start_load:.1f} s",
        )

        # Preprocess all images in one batch
        processed_images = preprocess_images(images, model_cfg['input_size'])

        # Predict segmentation masks
        for i, (img, mask) in tqdm(
            enumerate(zip(processed_images, masks)),
            total=len(images),
            desc=f'Segmentation of {class_name}',
            unit='image',
        ):
            predict_mask = model.predict(images=np.array([img]), device=device)[0]
            resized_mask = cv2.resize(
                predict_mask,
                tuple(output_size),
                interpolation=cv2.INTER_NEAREST,
            )
            if resized_mask.ndim > 2:
                resized_mask = resized_mask[:, :, class_meta['index']]
            class_idx = CLASS_IDS[class_name] - 1  # type: ignore
            mask[:, :, class_idx] = resized_mask
    return masks


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main function to perform histology image segmentation prediction."""
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Setup paths and device
    device = pick_device(option=cfg.device)
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    models_dir = str(os.path.join(PROJECT_DIR, cfg.models_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # Load dataset
    start = time.time()
    images, masks, images_name = data_processing(
        data_path=data_dir,
        save_dir=save_dir,
        output_size=cfg.output_size,
    )
    log.info(f'Number of images: {len(images_name)}')

    # Perform inference
    start_inference = time.time()
    masks = segment(
        images=images,
        masks=masks,
        output_size=cfg.output_size,
        classes=cfg.classes,
        models_dir=models_dir,
        device=device,
    )
    log.info(f'Prediction time: {time.time() - start_inference:.1f} s')

    # Save results
    save_results(
        images=images,
        masks=masks,
        images_name=images_name,
        classes=cfg.classes,
        save_dir=save_dir,
    )
    log.info(f'Overall computation time: {time.time() - start:.1f} s')
    log.info('Complete')


if __name__ == '__main__':
    main()
