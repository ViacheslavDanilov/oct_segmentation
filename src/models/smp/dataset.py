import logging
import multiprocessing
import os
import random
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.utils import CLASS_ID


class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        input_size: int = 512,
        use_augmentation: bool = False,
    ):
        self.classes = classes
        mask_paths = glob(f'{data_dir}/mask/*.[pj][np][pge]')
        self.input_size = input_size

        num_cores = multiprocessing.cpu_count()
        check_list = Parallel(n_jobs=num_cores, backend='threading')(
            delayed(self.data_check)(f'{data_dir}/img', mask_id)
            for mask_id in tqdm(mask_paths, desc='image load')
        )

        self.img_paths = list(np.array(check_list)[:, 1])
        self.mask_paths = list(np.array(check_list)[:, 0])
        self.class_values = [CLASS_ID[cl] for _, cl in enumerate(self.classes)]

        self.use_augmentation = use_augmentation

    def __getitem__(self, i: int):
        img = cv2.imread(self.img_paths[i])
        img = cv2.resize(img, (self.input_size, self.input_size))
        mask = cv2.imread(self.mask_paths[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.use_augmentation:
            transform = self.get_img_augmentation(input_size=self.input_size)
            sample = transform(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        img, mask = self.to_tensor_shape(img), self.to_tensor_shape(mask)

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def data_check(
        img_dir: str,
        ann_id: str,
    ) -> Union[Tuple[str, str], None]:
        img_name = Path(ann_id).name
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            return ann_id, img_path
        else:
            logging.warning(f'Img path: {img_path} not exist')
            return None

    @staticmethod
    def to_tensor_shape(
        x: np.ndarray,
    ) -> np.ndarray:
        return x.transpose([2, 0, 1]).astype('float32')

    @staticmethod
    def get_img_augmentation(
        input_size: int,
    ) -> albu.Compose:
        transform = [
            albu.HorizontalFlip(
                p=0.50,
            ),
            albu.ShiftScaleRotate(
                p=0.20,
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
            ),
            albu.RandomCrop(
                p=0.2,
                height=int(random.uniform(0.8, 0.9) * input_size),
                width=int(random.uniform(0.8, 0.9) * input_size),
            ),
            albu.PadIfNeeded(
                p=1.0,
                min_height=input_size,
                min_width=input_size,
                always_apply=True,
                border_mode=0,
            ),
            albu.GaussNoise(
                p=0.20,
                var_limit=(3.0, 10.0),
            ),
            albu.Perspective(
                p=0.20,
                scale=(0.05, 0.1),
            ),
            albu.RandomBrightnessContrast(
                p=0.20,
                brightness_limit=0.2,
                contrast_limit=0.2,
            ),
            albu.HueSaturationValue(
                p=0.20,
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
            ),
        ]
        return albu.Compose(transform)


class OCTDataModule(pl.LightningDataModule):
    """A data module used to create training and validation dataloaders with OCT images."""

    def __init__(
        self,
        classes: List[str],
        input_size: int = 448,
        batch_size: int = 2,
        num_workers: int = 2,
        data_dir: str = 'data/final',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            self.train_dataloader_set = OCTDataset(
                input_size=self.input_size,
                data_dir=f'{self.data_dir}/train',
                classes=self.classes,
                use_augmentation=True,
            )
            self.val_dataloader_set = OCTDataset(
                input_size=self.input_size,
                data_dir=f'{self.data_dir}/test',
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
    dataset = OCTDataset(
        data_dir='data/final/train',
        classes=['Lipid core', 'Lumen', 'Fibrous cap', 'Vasa vasorum'],
        input_size=448,
        use_augmentation=False,
    )
    for i in range(30):
        img, mask = dataset[i]
    print('Complete')
