import os
from typing import List

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.data.utils import get_file_list


class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
        self,
        subset_dir: str,
        classes: List[str],
        input_size: int = 448,
        use_augmentation: bool = False,
    ):
        self.input_size = input_size
        self.classes = classes
        self.classes_idx = [idx + 1 for idx, _ in enumerate(self.classes)]
        self.samples = self.get_list_of_samples(subset_dir)
        self.use_augmentation = use_augmentation

    def __getitem__(
        self,
        idx: int,
    ):
        img_path, mask_path = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.classes_idx]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.use_augmentation:
            transform = self.get_img_augmentation(input_size=self.input_size)
            sample = transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = self.to_tensor(np.array(image)), self.to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_list_of_samples(
        data_dir: str,
    ) -> List[List[str]]:
        img_dir = os.path.join(data_dir, 'img')
        mask_dir = os.path.join(data_dir, 'mask')
        image_paths = get_file_list(src_dirs=img_dir, ext_list='.png')
        mask_paths = get_file_list(src_dirs=mask_dir, ext_list='.png')
        assert len(image_paths) == len(
            mask_paths,
        ), f'Number of images is not equal to the number of masks ({len(image_paths)} vs {len(mask_paths)})'

        sample_list = []

        for image_path in image_paths:
            mask_path = image_path.replace('img', 'mask')
            if mask_path in mask_paths:
                sample_list.append([image_path, mask_path])

        return sample_list

    @staticmethod
    def to_tensor(
        x: np.ndarray,
    ) -> np.ndarray:
        return x.transpose([2, 0, 1]).astype('float32')

    @staticmethod
    def get_img_augmentation(
        input_size: int,
    ) -> albu.Compose:
        transform = [
            albu.HorizontalFlip(
                p=0.5,
            ),  # TODO: sure about the aggressiveness of this augmentation pipeline?
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
                height=input_size,  # TODO: random(0.7; 0.9)*input_size, seed (https://github.com/open-mmlab/mmdetection/issues/2558)
                width=input_size,  # TODO: random(0.7; 0.9)*input_size, seed (https://github.com/open-mmlab/mmdetection/issues/2558)
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


class OCTDataModule(pl.LightningDataModule):
    """A data module used to create training and validation dataloaders with OCT images."""

    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        input_size: int = 448,
        batch_size: int = 2,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
        self,
        stage: str = 'fit',
    ):
        if stage == 'fit':
            self.train_dataloader_set = OCTDataset(
                input_size=self.input_size,
                subset_dir=f'{self.data_dir}/train',
                classes=self.classes,
                use_augmentation=True,
            )
            self.val_dataloader_set = OCTDataset(
                input_size=self.input_size,
                subset_dir=f'{self.data_dir}/test',
                classes=self.classes,
                use_augmentation=False,
            )
        elif stage == 'test':
            raise ValueError('The "test" method is not yet implemented')
        else:
            raise ValueError(f'Unsupported stage value: {stage}')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataloader_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataloader_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    oct_dataset = OCTDataset(
        subset_dir='data/final/train',
        classes=['Lipid core', 'Lumen', 'Fibrous cap', 'Vasa vasorum'],
        input_size=448,
        use_augmentation=False,
    )
    sample = oct_dataset[0]
    print('Complete')
