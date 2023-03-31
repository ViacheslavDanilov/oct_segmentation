import os
import hydra
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import wandb
import mlflow
import torch
from joblib import Parallel, delayed
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import List
from pytorch_lightning import loggers
from segmentation_models_pytorch.encoders import get_preprocessing_fn

os.environ['WANDB_API_KEY'] = '0a94ef68f2a7a8b709671d6ef76e61580d20da7f'

DEFAULT_CLASSES = ['Lipid core', 'Lumen', 'Fibrous cap', 'Vasa vasorum']


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


# def get_item(
#         image_path: str,
#         mask_path: str,
#         input_size: int,
#         class_values: List[int],
# ):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (input_size, input_size))
#     mask = cv2.imread(mask_path, 0)
#     mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
#
#     masks = [(mask == v) for v in class_values]
#     mask = np.stack(masks, axis=-1).astype('float')
#
#     return image, mask
#
#
# def get_data(
#         input_size,
#         ann_dir,
#         img_dir,
#         classes=None,
# ):
#     num_cores = multiprocessing.cpu_count()
#     check_list = Parallel(n_jobs=num_cores, backend="threading") \
#         (delayed(data_checked)(img_dir, ann_id) for ann_id in tqdm(glob(f'{ann_dir}/*.png'), desc='image load'))
#
#     images_fps = list(np.array(check_list)[:, 1])
#     masks_fps = list(np.array(check_list)[:, 0])
#     class_values = [DEFAULT_CLASSES.index(cls) + 1 for cls in classes]
#
#     dataset = Parallel(n_jobs=num_cores, backend="threading") \
#         (delayed(get_item)(img_path, mask_path, input_size, class_values) for img_path, mask_path in tqdm(zip(images_fps, masks_fps), desc='image load'))
#
#     return np.array(dataset)[:, 0], np.array(dataset)[:, 1]


class CustomImageDataset(Dataset):
    def __init__(
            self,
            input_size,
            ann_dir,
            img_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.classes = classes
        self.ids = glob(f'{ann_dir}/*.png')
        self.input_size = input_size

        num_cores = multiprocessing.cpu_count()
        check_list = Parallel(n_jobs=num_cores, backend="threading")\
            (delayed(data_checked)(img_dir, ann_id) for ann_id in tqdm(self.ids, desc='image load'))

        self.images_fps = list(np.array(check_list)[:, 1])
        self.masks_fps = list(np.array(check_list)[:, 0])

        self.class_values = [DEFAULT_CLASSES.index(cls) + 1 for cls in self.classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(mask, (self.input_size, self.input_size))

        # mask[mask != 0] = 2

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        else:
            image, mask = to_tensor(np.array(image)), to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.ids)


class MyModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.classes = out_classes
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # image = batch["image"]
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # mask = batch["mask"]
        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        # assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        # assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel", num_classes=self.classes)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        wandb.log(metrics)
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # def on_train_epoch_end(self):
    #     return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    # def on_validation_epoch_end(self):
    #     return self.shared_epoch_end(outputs, "valid")

    # def test_step(self, batch, batch_idx):
    #     return self.shared_step(batch, "test")
    #
    # def test_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='training_models_smt',
    version_base=None,
)
def main(
        cfg: DictConfig,
):
    model_dir = 'models/model_2'
    # model = smp.Unet(
    #     encoder_name="resnet34",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=3,
    # )
    # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    train_dataset = CustomImageDataset(
        input_size=224,
        ann_dir=cfg.ann_dir,
        img_dir=cfg.img_dir,
        classes=cfg.classes,
    )

    valid_dataset = CustomImageDataset(
        input_size=224,
        ann_dir=cfg.ann_dir,
        img_dir=cfg.img_dir,
        classes=cfg.classes,
    )

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    model = MyModel("FPN", "resnet34", in_channels=3, out_classes=len(cfg.classes))

    trainer = pl.Trainer(
        # gpus=0,
        devices=-1,
        accelerator='cuda',
        max_epochs=5,
        logger=loggers.WandbLogger(
                save_dir=model_dir,
            ),
        default_root_dir=model_dir,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == '__main__':
    main()