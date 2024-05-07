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
from src.models.smp.utils import calculate_dice, calculate_iou, pick_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_images(
    images: List[np.ndarray],
    image_names: List[str],
    output_size: Tuple[int, int],
    save_dir: str,
    num_colors_threshold: int = 10,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for image, image_name in zip(images, image_names):
        image_name = image_name.replace(' ', '_')

        # Calculate the number of unique colors in the image
        flat_image = image.reshape(-1, 3)
        unique_colors = np.unique(flat_image, axis=0)
        num_unique_colors = len(unique_colors)

        # Choose interpolation method based on the number of unique colors
        interpolation = (
            cv2.INTER_NEAREST if num_unique_colors <= num_colors_threshold else cv2.INTER_LANCZOS4
        )

        # Resize and save the image
        if output_size:
            image = cv2.resize(image, output_size, interpolation=interpolation)
        cv2.imwrite(os.path.join(save_dir, image_name), image)


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
    for class_name in class_names:
        metrics[class_name] = {
            'Dice': 0.0,
            'IOU': 0.0,
        }
    input_size = (model_cfg['input_size'],) * 2

    # Extract activation maps and save with overlay images
    # img_paths = img_paths[:1]  # FIXME: used only for debugging
    for img_path in tqdm(img_paths, desc='Extract and save activation maps', unit='image'):
        img = cv2.imread(img_path)
        img = cv2.resize(img, input_size)
        mask_pred = model.predict(images=np.array([img]), device=device)[0]

        img_stem = Path(img_path).stem
        mask_gt_path = os.path.join(mask_dir, f'{img_stem}.tiff')
        mask_gt = tifffile.imread(mask_gt_path)

        for class_idx in range(len(class_names)):
            class_mask_pred = mask_pred[:, :, class_idx]
            targets = cam_processor.get_targets(
                class_idx=class_idx,
                class_mask=class_mask_pred,
            )
            mask_cam = cam_processor.extract_activation_map(
                image=img,
                targets=targets,
                aug_smooth=cfg.aug_smooth,
                eigen_smooth=cfg.eigen_smooth,
            )
            pred_mask = mask_cam.copy()
            pred_mask[pred_mask > cfg.map_threch] = 255
            pred_mask[pred_mask != 255] = 0

            metrics[class_names[class_idx]]['Dice'] += calculate_dice(
                pred_mask=cv2.resize(pred_mask, (mask_gt.shape[0], mask_gt.shape[1])),
                gt_mask=mask_gt[:, :, class_idx],
            )
            metrics[class_names[class_idx]]['IOU'] += calculate_iou(
                pred_mask=cv2.resize(pred_mask, (mask_gt.shape[0], mask_gt.shape[1])),
                gt_mask=mask_gt[:, :, class_idx],
            )

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

            save_images(
                images=[img, img_cam, color_class_mask_pred, color_class_mask_gt],
                image_names=[
                    f'{img_stem}_input.png',
                    f'{img_stem}_{class_name}_{cfg.cam_method}.png',
                    f'{img_stem}_{class_name}_pred.png',
                    f'{img_stem}_{class_name}_gt.png',
                ],
                output_size=cfg.output_size,
                save_dir=os.path.join(save_dir, model_cfg['model_name']),
            )

    # # TODO: Incompatible types in assignment (expression has type "float", target has type "int")
    for class_name in class_names:
        metrics[class_name]['IOU'] /= len(img_paths)
        metrics[class_name]['Dice'] /= len(img_paths)
    log.info(f'Metrics: {metrics}')
    with open('metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    log.info('Complete!')


if __name__ == '__main__':
    main()
