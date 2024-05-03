import json
import logging
import os
from glob import glob

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import CLASS_IDS_REVERSED
from src.models.cam_processor import CAMProcessor
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import pick_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
    target_layers = [model.model.encoder.layer4[-1]]
    class_names = model_cfg['classes']

    img_dir = os.path.join(data_dir, 'img')
    mask_dir = os.path.join(data_dir, 'mask')
    img_paths = glob(os.path.join(img_dir, '*.png'))
    img_paths = img_paths[:1]  # TODO: remove this line

    cam_processor = CAMProcessor(
        model=model,
        cam_method=cfg.cam_method,
        device=device,
        target_layers=target_layers,
    )
    for img_path in tqdm(img_paths, desc='Process images', unit='image'):
        img = cv2.imread(img_path)
        output_size = (model_cfg['input_size'],) * 2
        img = cv2.resize(img, output_size)
        mask = model.predict(images=np.array([img]), device=device)[0]

        # img_stem = Path(img_path).stem
        # map_dir = os.path.join(save_dir, model_cfg['model_name'])
        # os.makedirs(map_dir, exist_ok=True)
        #
        # mask_gt_path = os.path.join(mask_dir, f'{img_stem}.tiff')
        # mask_gt = tifffile.imread(mask_gt_path)
        # mask_gt = cv2.resize(mask_gt, cfg.output_size, interpolation=cv2.INTER_NEAREST)

        for class_idx in range(len(class_names)):
            class_name = CLASS_IDS_REVERSED[class_idx + 1]
            class_mask = mask[:, :, class_idx]
            img_cam_gray = cam_processor.process_image(
                img=img,
                class_idx=class_idx,
                class_mask=class_mask,
            )

            # img_rgb = Image.open(img_path).resize(
            #     (model_cfg['input_size'], model_cfg['input_size']),
            # )
            # img_rgb = np.float32(img_rgb) / 255
            # img_rgb = np.array(img_rgb)

            # map_name = f'{img_stem}_{class_name}_{cfg.cam_method}.png'

            # with cam_method(model=model, target_layers=target_layers) as cam:
            #     # init source image
            #     source_image = Image.open(img_path)
            #     source_image = source_image.resize(cfg.output_size)
            #     # init cam image
            #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            #

    # for img_path in tqdm(img_paths, desc='Process images', unit='image'):
    #     img = preprocessing_img(img_path, input_size=model_cfg['input_size'])
    #     mask = model.predict(images=np.array([img]), device=device)[0]
    #
    #     img_stem = Path(img_path).stem
    #     map_dir = os.path.join(save_dir, model_cfg['model_name'])
    #     os.makedirs(map_dir, exist_ok=True)
    #
    #     mask_gt_path = os.path.join(mask_dir, f'{img_stem}.tiff')
    #     mask_gt = tifffile.imread(mask_gt_path)
    #     mask_gt = cv2.resize(mask_gt, cfg.output_size, interpolation=cv2.INTER_NEAREST)
    #
    #     for class_idx in range(len(class_names)):
    #         class_name = CLASS_IDS_REVERSED[class_idx + 1]
    #         mask_class = np.float32(np.array(mask[:, :, class_idx]).astype(bool))
    #         input_tensor = torch.Tensor(np.array(img)).to(device)
    #         targets = [SemanticSegmentationTarget(class_idx, mask_class)]
    #
    #         a = cam_processor.process_image_with_cam(
    #             img_path=img_path,
    #             output_size=cfg.output_size,
    #             targets=targets,
    #         )
    #
    #         img_rgb = Image.open(img_path).resize((model_cfg['input_size'], model_cfg['input_size']))
    #         img_rgb = np.float32(img_rgb) / 255
    #         img_rgb = np.array(img_rgb)
    #
    #         cam_methods = {
    #             'GradCAM': GradCAM,
    #             'HiResCAM': HiResCAM,
    #             'GradCAMElementWise': GradCAMElementWise,
    #             'GradCAMPlusPlus': GradCAMPlusPlus,
    #             'XGradCAM': XGradCAM,
    #             'AblationCAM': AblationCAM,
    #             'EigenCAM': EigenCAM,
    #             'EigenGradCAM': EigenGradCAM,
    #             'LayerCAM': LayerCAM,
    #         }
    #
    #         map_name = f'{img_stem}_{class_name}_{cfg.cam_method}.png'
    #         cam_method = cam_methods[cfg.cam_method]
    #         # from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, \
    #         #     ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
    #         with cam_method(model=model, target_layers=target_layers) as cam:
    #             # init source image
    #             source_image = Image.open(img_path)
    #             source_image = source_image.resize(cfg.output_size)
    #             # init cam image
    #             grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    #             cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    #             cam_image = Image.fromarray(cam_image).resize(cfg.output_size)
    #             # init class color img
    #             color = Image.new('RGB', cam_image.size, CLASS_COLORS_RGB[class_name])
    #             # init gr
    #             color_mask_gt = Image.new('RGB', cam_image.size, (128, 128, 128))
    #             color_mask_gt.paste(
    #                 color,
    #                 (0, 0),
    #                 Image.fromarray(mask_gt[:, :, class_idx]).resize(cfg.output_size),
    #             )
    #             # init predict
    #             color_mask = Image.new('RGB', cam_image.size, (128, 128, 128))
    #             color_mask.paste(
    #                 color,
    #                 (0, 0),
    #                 Image.fromarray(np.array(mask_class * 255).astype('uint8')).resize(
    #                     cfg.output_size,
    #                 ),
    #             )
    #             # init and save result img
    #             output = Image.new('RGB', (cam_image.size[0] * 4, cam_image.size[1]), (0, 0, 0))
    #             output.paste(source_image, (0, 0))
    #             output.paste(cam_image, (cam_image.size[0], 0))
    #             output.paste(color_mask_gt, (cam_image.size[0] * 2, 0))
    #             output.paste(color_mask, (cam_image.size[0] * 3, 0))
    #             output.save(os.path.join(map_dir, map_name), quality=100)


if __name__ == '__main__':
    main()
