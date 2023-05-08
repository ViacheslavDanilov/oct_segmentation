import multiprocessing
import os
from glob import glob

import albumentations as albu
import cv2
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm


def data_checked(
        img_dir: str,
        ann_id: str,
):
    img_path = f'{img_dir}/{os.path.basename(ann_id)}'
    if os.path.exists(img_path):
        return ann_id, img_path
    else:
        print(f'Warning: not exists {img_path}')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_img_augmentation(input_size):
    transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.35,
            rotate_limit=45,
            shift_limit=0.1,
            p=0.8,
            border_mode=0,
        ),
        albu.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            always_apply=True,
            border_mode=0,
        ),
        albu.RandomCrop(
            height=input_size,
            width=input_size,
            always_apply=True,
            p=0.5,
        ),
        albu.GaussNoise(p=0.25),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(transform)


class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
            self,
            input_size,
            img_dir,
            mask_dir,
            classes=None,
            augmentation=None,
    ):
        self.classes = classes
        self.ids = glob(f'{mask_dir}/*.png')
        self.input_size = input_size

        num_cores = multiprocessing.cpu_count()
        check_list = Parallel(n_jobs=num_cores, backend='threading')(
            delayed(data_checked)(img_dir, mask_id) for mask_id in tqdm(self.ids, desc='image load')
        )

        self.images_fps = list(np.array(check_list)[:, 1])
        self.masks_fps = list(np.array(check_list)[:, 0])
        self.class_values = [idx + 1 for idx, _ in enumerate(self.classes)]

        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = to_tensor(np.array(image)), to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.ids)
