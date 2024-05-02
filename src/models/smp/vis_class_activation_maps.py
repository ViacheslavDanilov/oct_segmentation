import json
import os.path
from glob import glob

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS_REVERSED
from src.models.smp.model import OCTSegmentationModel


def to_tensor(
        x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


def preprocessing_img(
        img_path: str,
        input_size: int,
):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (input_size, input_size))
    image = to_tensor(np.array(image))
    return image


if __name__ == '__main__':
    model_path = '/home/vladislav/MaNet_resnet50'
    data_path = 'data/visualization/img'
    save_dir = 'data/act_map'

    for img_path in glob(f'{data_path}/*.png'):
        img_name = os.path.basename(img_path)
        with open(f'{model_path}/config.json', 'r') as file:
            model_cfg = json.load(file)
        model = OCTSegmentationModel.load_from_checkpoint(
            checkpoint_path=f'{model_path}/weights.ckpt',
            encoder_weights=None,
            arch=model_cfg['architecture'],
            encoder_name=model_cfg['encoder'],
            model_name=model_cfg['model_name'],
            in_channels=3,
            classes=model_cfg['classes'],
            map_location='cuda:0',
        )
        model.eval()
        image = preprocessing_img(
            img_path,
            input_size=model_cfg['input_size'],
        )

        masks = model.predict(
            images=np.array([image]),
            device='cuda',
        )[0]

        from pytorch_grad_cam import GradCAM


        class SemanticSegmentationTarget:
            def __init__(self, category, mask):
                self.category = category
                self.mask = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    self.mask = self.mask.cuda()

            def __call__(self, model_output):
                return (model_output[self.category, :, :] * self.mask).sum()


        target_layers = [model.model.encoder.layer4[-1]]

        if not os.path.exists(f'{save_dir}/{img_name.split(".")[0]}'):
            os.makedirs(f'{save_dir}/{img_name.split(".")[0]}')

        mask_gr = tifffile.imread(f"{img_path.replace('img', 'mask').split('.')[0]}.tiff")
        mask_gr = cv2.resize(
            mask_gr,
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST,
        )

        for class_detection in range(3):
            class_name = CLASS_IDS_REVERSED[class_detection + 1]
            mask_class = np.float32(np.array(masks[:, :, class_detection]).astype(bool))
            targets = [SemanticSegmentationTarget(class_detection, mask_class)]
            input_tensor = image.copy()
            input_tensor = torch.Tensor(np.array(input_tensor)).to('cuda')

            rgb_img = Image.open(img_path).resize(
                (model_cfg['input_size'], model_cfg['input_size']),
            )
            rgb_img = np.float32(rgb_img) / 255
            rgb_img = np.array(rgb_img)

            with GradCAM(
                    model=model,
                    target_layers=target_layers,
                    # use_cuda=torch.cuda.is_available()
            ) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets,
                )[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
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
                    Image.fromarray(mask_gr[:, :, class_detection]).resize((1024, 1024)),
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

                output.save(
                    f'{save_dir}/{img_name.split(".")[0]}/{class_name}.png',
                )
                # output.show()
