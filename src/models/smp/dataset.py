import logging
import os
import random
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
import tifffile
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.utils import CLASS_IDS


class OCTDataModule(pl.LightningDataModule):
    """A data module used to create training and validation dataloaders with OCT images."""

    def __init__(
        self,
        classes: List[str],
        data_dir: str = 'data/cv/fold_1',
        input_size: int = 512,
        batch_size: int = 2,
        num_workers: int = 2,
        use_augmentation: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            self.train_dataloader_set = OCTDataset(
                input_size=self.input_size,
                data_dir=f'{self.data_dir}/train',
                classes=self.classes,
                use_augmentation=self.use_augmentation,
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
        self.class_ids = [CLASS_IDS[cl] for _, cl in enumerate(self.classes)]
        self.input_size = input_size
        self.use_augmentation = use_augmentation

        mask_paths = glob(os.path.join(data_dir, 'mask', '*.tiff'))
        num_jobs = int(os.cpu_count() * 0.5)
        pair_list = Parallel(n_jobs=num_jobs)(
            delayed(self.verify_pairs)(
                img_dir=os.path.join(data_dir, 'img'),
                mask_path=mask_path,
                class_ids=self.class_ids,
            )
            for mask_path in tqdm(mask_paths, desc='Check image-mask pairs')
        )
        pair_list = [pair for pair in pair_list if pair is not None]
        if not pair_list:
            raise ValueError('Warning: No correct data found')
        print(f'Number of image-mask pairs: {len(pair_list)}')

        self.img_paths, self.mask_paths = zip(*pair_list)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, (self.input_size, self.input_size))
        mask = tifffile.imread(self.mask_paths[idx])
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = []
        for class_id in self.class_ids:
            channel_id = class_id - 1  # type: ignore
            masks.append(np.array(mask[:, :, channel_id], dtype='bool'))
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
    def verify_pairs(
        img_dir: str,
        mask_path: str,
        class_ids: List[int],
    ) -> Union[Tuple[str, str], None]:
        mask = tifffile.imread(mask_path)
        img_name = Path(mask_path).stem
        img_path = os.path.join(img_dir, f'{img_name}.png')

        if not os.path.exists(img_path):
            logging.warning(f'Image: {img_path} does not exist')
            return None

        for class_id in class_ids:
            channel_id = class_id - 1
            unique_values = np.unique(mask[:, :, channel_id])
            if np.any(unique_values > 1):
                return img_path, mask_path

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


if __name__ == '__main__':
    dataset = OCTDataset(
        data_dir='data/cv_dev/fold_1/train',
        classes=['Lumen', 'Fibrous cap', 'Lipid core', 'Vasa vasorum'],
        input_size=512,
        use_augmentation=False,
    )
    for idx in range(30):
        img, mask = dataset[idx]
    print('Complete')
