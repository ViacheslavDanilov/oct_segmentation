import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import tifffile
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.convert_int_to_cv import colorize_mask
from src.data.utils import CLASS_IDS_REVERSED
from src.models.cam_processor import CAMProcessor
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import pick_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def combine_images(
    images: List[np.ndarray],
    output_size: Tuple[int, int],
) -> np.ndarray:
    resized_images = [
        cv2.resize(img, output_size, interpolation=cv2.INTER_NEAREST) for img in images
    ]
    combined_image = np.hstack(resized_images)
    return combined_image


def save_image(
    image: np.ndarray,
    image_name: str,
    save_dir: str,
) -> None:
    image_name = image_name.replace(' ', '_')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='visualize_activation_maps',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    model_dir = str(os.path.join(PROJECT_DIR, cfg.model_dir))
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    # Initialize model
    device = pick_device(cfg.device)
    with open(os.path.join(model_dir, 'config.json'), 'r') as file:
        model_cfg = json.load(file)
    model = OCTSegmentationModel.load_from_checkpoint(
        checkpoint_path=os.path.join(model_dir, 'weights.ckpt'),
        encoder_weights=None,
        arch=model_cfg['architecture'],
        encoder_name=model_cfg['encoder'],
        model_name=model_cfg['model_name'],
        in_channels=3,
        classes=model_cfg['classes'],
        map_location=device,
    )
    model.eval()
    target_layers = [model.model.encoder.layer4[-1]]

    # Initialize CAM processor
    cam_processor = CAMProcessor(
        model=model,
        cam_method=cfg.cam_method,
        device=device,
        target_layers=target_layers,
        percentile=10,
    )

    # Define additional parameters
    img_dir = os.path.join(data_dir, 'img')
    mask_dir = os.path.join(data_dir, 'mask')
    img_paths = glob(os.path.join(img_dir, '*.png'))
    class_names = model_cfg['classes']
    metrics = {}
    for cl in class_names:
        metrics[cl] = {
            'confidence increase when removing 25%': 0,
            'confidence increase percent': 0,
        }
    input_size = (model_cfg['input_size'],) * 2

    # Extract activation maps and save with overlay images
    for img_path in tqdm(img_paths, desc='Save activation maps', unit='image'):
        img = cv2.imread(img_path)
        img = cv2.resize(img, input_size)
        mask_pred = model.predict(images=np.array([img]), device=device)[0]

        img_stem = Path(img_path).stem
        mask_gt_path = os.path.join(mask_dir, f'{img_stem}.tiff')
        mask_gt = tifffile.imread(mask_gt_path)

        for class_idx in range(len(class_names)):
            class_mask_pred = mask_pred[:, :, class_idx]
            mask_cam = cam_processor.extract_activation_map(
                image=img,
                class_idx=class_idx,
                class_mask=class_mask_pred,
            )
            scores = cam_processor.compute_metrics(
                image=img,
                mask=mask_cam,
                class_idx=class_idx,
                class_mask=class_mask_pred,
            )
            metrics[class_names[class_idx]]['confidence increase when removing 25%'] += scores[0]
            metrics[class_names[class_idx]]['confidence increase percent'] += scores[1]

            img_cam = cam_processor.overlay_activation_map(
                image=img,
                mask=mask_cam,
                image_weight=0.5,
            )

            class_name = CLASS_IDS_REVERSED[class_idx + 1]
            color_class_mask_gt = colorize_mask(
                mask=mask_gt,
                classes=[class_name],
            )
            color_class_mask_gt = cv2.cvtColor(color_class_mask_gt, cv2.COLOR_BGR2RGB)

            color_class_mask_pred = colorize_mask(
                mask=(mask_pred * 255).astype('uint8'),
                classes=[class_name],
            )
            color_class_mask_pred = cv2.cvtColor(color_class_mask_pred, cv2.COLOR_BGR2RGB)

            img_stack = combine_images(
                images=[img, img_cam, color_class_mask_pred, color_class_mask_gt],
                output_size=cfg.output_size,
            )

            save_image(
                image=img_stack,
                image_name=f'{img_stem}_{class_name}_{cfg.cam_method}.png',
                save_dir=os.path.join(save_dir, model_cfg['model_name']),
            )
            print('')
    for cl in class_names:
        metrics[cl]['confidence increase when removing 25%'] /= len(img_paths)
        metrics[cl]['confidence increase percent'] /= len(img_paths)
    print(metrics)
    with open('metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)


if __name__ == '__main__':
    main()
