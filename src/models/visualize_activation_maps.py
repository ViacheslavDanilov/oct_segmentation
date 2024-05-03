import json
import logging
import os
from glob import glob
from pathlib import Path

import cv2
import hydra
import numpy as np
import tifffile
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS_REVERSED
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import pick_device, preprocessing_img

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SemanticSegmentationTarget:
    """Represents a semantic segmentation target.

    Attributes:
        category (int): The category of the target.
        mask (torch.Tensor): The mask associated with the target.

    Methods:
        __init__: Initializes a SemanticSegmentationTarget instance.
        __call__: Computes the target based on the model output.
    """

    def __init__(self, category, mask):
        self.category = category
        self.mask = (
            torch.from_numpy(mask).cuda() if torch.cuda.is_available() else torch.from_numpy(mask)
        )

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


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
    class_names = model_cfg['classes']

    img_dir = os.path.join(data_dir, 'img')
    mask_dir = os.path.join(data_dir, 'mask')
    img_paths = glob(os.path.join(img_dir, '*.png'))
    for img_path in tqdm(img_paths, desc='Process images', unit='image'):
        img = preprocessing_img(img_path, input_size=model_cfg['input_size'])
        mask = model.predict(images=np.array([img]), device=device)[0]
        target_layers = [model.model.encoder.layer4[-1]]

        img_stem = Path(img_path).stem
        map_dir = os.path.join(save_dir, model_cfg['model_name'])
        os.makedirs(map_dir, exist_ok=True)

        mask_gt_path = os.path.join(mask_dir, f'{img_stem}.tiff')
        mask_gt = tifffile.imread(mask_gt_path)
        mask_gt = cv2.resize(mask_gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        for class_detection in range(len(class_names)):
            class_name = CLASS_IDS_REVERSED[class_detection + 1]
            mask_class = np.float32(np.array(mask[:, :, class_detection]).astype(bool))
            targets = [SemanticSegmentationTarget(class_detection, mask_class)]

            input_tensor = torch.Tensor(np.array(img)).to(device)

            img_rgb = Image.open(img_path).resize(
                (model_cfg['input_size'], model_cfg['input_size']),
            )
            img_rgb = np.float32(img_rgb) / 255
            img_rgb = np.array(img_rgb)  # TODO: это BGR изображение?

            map_name = f'{img_stem}_{class_name}.png'
            # TODO: add input image to the beginning i.e. input image - cam - GT - Predict
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
                cam_image = Image.fromarray(cam_image).resize((1024, 1024))
                output = Image.new('RGB', (cam_image.size[0] * 3, cam_image.size[1]), (0, 0, 0))

                # source_image = Image.open(img_path)
                # source_image = source_image.resize((1024, 1024))
                # output.paste(source_image, (0, 0))
                #
                # output.paste(cam_image, (source_image.size[0], 0))
                # color_mask = Image.new('RGB', cam_image.size, (128, 128, 128))
                # color_mask.paste(
                #     Image.new('RGB', cam_image.size, CLASS_COLORS_RGB[class_name]),
                #     (0, 0),
                #     Image.fromarray(np.array(mask_class * 255).astype('uint8')).resize((1024, 1024))
                # )
                # output.paste(color_mask, (source_image.size[0] * 2, 0))
                color = Image.new('RGB', cam_image.size, CLASS_COLORS_RGB[class_name])

                output.paste(cam_image, (0, 0))

                color_mask_gt = Image.new('RGB', cam_image.size, (128, 128, 128))
                color_mask_gt.paste(
                    color,
                    (0, 0),
                    Image.fromarray(mask_gt[:, :, class_detection]).resize((1024, 1024)),
                )
                output.paste(color_mask_gt, (cam_image.size[0], 0))

                color_mask = Image.new('RGB', cam_image.size, (128, 128, 128))
                color_mask.paste(
                    color,
                    (0, 0),
                    Image.fromarray(np.array(mask_class * 255).astype('uint8')).resize(
                        (1024, 1024),
                    ),
                )
                output.paste(color_mask, (cam_image.size[0] * 2, 0))

                output.save(os.path.join(map_dir, map_name), quality=100)


if __name__ == '__main__':
    main()
