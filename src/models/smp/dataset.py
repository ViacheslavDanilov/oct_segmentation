import logging
import multiprocessing
import os
from glob import glob
from typing import List, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
from clearml import Dataset as cl_dataset
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        input_size: int = 224,
        use_augmentation: bool = False,
    ):
        self.classes = classes
        self.ids = glob(f'{data_dir}/mask/*.png')
        self.input_size = input_size

        num_cores = multiprocessing.cpu_count()
        check_list = Parallel(n_jobs=num_cores, backend='threading')(
            delayed(self.data_check)(f'{data_dir}/img', mask_id)
            for mask_id in tqdm(self.ids, desc='image load')
        )

        self.images_idx = list(np.array(check_list)[:, 1])
        self.masks_idx = list(np.array(check_list)[:, 0])
        self.class_values = [idx + 1 for idx, _ in enumerate(self.classes)]

        self.use_augmentation = use_augmentation

    def __getitem__(self, i: int):
        image = cv2.imread(self.images_idx[i])
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.masks_idx[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.use_augmentation:
            transform = self.get_img_augmentation(input_size=self.input_size)
            sample = transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = self.to_tensor(np.array(image)), self.to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.images_idx)

    @staticmethod
    def data_check(
        img_dir: str,
        ann_id: str,
    ) -> Union[Tuple[str, str], None]:
        img_path = f'{img_dir}/{os.path.basename(ann_id)}'
        if os.path.exists(img_path):
            return ann_id, img_path
        else:
            logging.warning(f'Img path: {img_path} not exist')
            return None

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
        dataset_name: str,  # TODO: we need to set data_dir instead of dataset_name
        classes: List[str],
        project_name: str = 'OCT segmentation',  # TODO: project_name is no longer needed when using ClearMLDataProcessor
        input_size: int = 224,
        batch_size: int = 2,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir = None
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    # TODO: What if I don't have the dataset in the cloud? What if I am running the training for the first time?
    # TODO: Move this method to the ClearMLDataProcessor class
    def prepare_data(self):
        dataset_clearml = cl_dataset.get(
            dataset_name=self.dataset_name,
            dataset_project=self.project_name,
        )
        # TODO: it is a read-only copy, while a mutable copy is needed
        # TODO: get_mutable_local_copy() should be used here
        self.data_dir = dataset_clearml.get_local_copy()

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
                data_dir=f'{self.data_dir}/val',
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


class ClearMLDataProcessor:
    """A class for processing data in the ClearML framework."""

    def __init__(
        self,
        data_dir: str,
    ):
        self.data_dir = data_dir

    def prepare_data(self):
        # TODO: if local data is newer than server data, upload it
        self.upload_dataset()

        # TODO: if no data is available locally, download it from the server
        self.download_dataset()

    @staticmethod
    def upload_dataset():
        print('Uploading...')

    @staticmethod
    def download_dataset():
        print('Downloading...')


# TODO: remove once classes are implemented and debugged
if __name__ == '__main__':
    oct_data_module = OCTDataModule(
        project_name='OCT segmentation',
        dataset_name='smp_dataset',
        classes=[
            'Lipid core',
            'Lumen',
            'Fibrous cap',
            'Vasa vasorum',
        ],
        input_size=224,
        batch_size=2,
        num_workers=2,
    )
    oct_data_module.prepare_data()
    print('Complete')
