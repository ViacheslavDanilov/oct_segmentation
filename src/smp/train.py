import logging
import multiprocessing
import os
from glob import glob

import cv2
import hydra
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from clearml import Dataset as ds
from clearml import Logger, Task
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEFAULT_CLASSES = (
    [  # TODO: why do we use it here if we have "classes" argument in the train config file?
        'Lipid core',
        'Lumen',
        'Fibrous cap',
        'Vasa vasorum',
    ]
)
COLOR = {
    'Lumen': (133, 21, 199),
    'Lipid core': (0, 252, 124),
    'Fibrous cap': (170, 178, 32),
    'Vasa vasorum': (34, 34, 178),
    'Artifact': (152, 251, 152),
}
EPOCH = 0


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


# TODO: this dataset should be moved to a separate file src/models/dataset.py
class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
        self,
        input_size,
        img_dir,
        mask_dir,
        classes=None,
        augmentation=None,  # TODO: augmentation should be implemented, but can be removed as an argument
        preprocessing=None,  # TODO: preprocessing can be removed as it is not utilized
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

        self.class_values = [DEFAULT_CLASSES.index(cls) + 1 for cls in self.classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # TODO: implement augmentation as an additional method of the class
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # TODO: I think we will never use the condigon under the if-loop section
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        else:
            image, mask = to_tensor(np.array(image)), to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.ids)


# TODO: the model should be moved to a separate file src/models/models.py
class OCTSegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of OCT images."""

    def __init__(self, arch, encoder_name, in_channels, classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=len(classes),
            **kwargs,
        )

        self.classes = classes
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode='multilabel',
            num_classes=len(self.classes),
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn)

        self.log('train/loss', loss, prog_bar=True, on_epoch=True)

        metrics = {
            f'train/IOU (mean)': iou.mean(),
        }
        for num, cl in enumerate(self.classes):
            metrics[f'train/IOU ({cl})'] = iou[:, num].mean()
        self.log_dict(metrics, on_epoch=True)

        return {
            'loss': loss,
        }

    def test_step(self, batch, batch_idx):
        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode='multilabel',
            num_classes=len(self.classes),
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn)
        metrics = {
            f'test/IOU (mean)': iou.mean(),
        }
        for num, cl in enumerate(self.classes):
            metrics[f'test/IOU ({cl})'] = iou[:, num].mean()
        self.log_dict(metrics, on_epoch=True)
        self.log('test/loss', loss, prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            global EPOCH
            logger = Logger.current_logger()
            img = img.permute(0, 2, 3, 1)
            img = img.squeeze().cpu().numpy().round()
            mask = mask.squeeze().cpu().numpy().round()
            pred_mask = pred_mask.squeeze().cpu().numpy().round()
            for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
                img_ = np.array(img_)
                img_g = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_p = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_0 = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                for cl, m, m_p in zip(self.classes, mask_, pr_mask):
                    # Groundtruth
                    img_g = cv2.addWeighted(
                        np.array(img_g).astype('uint8'),
                        1,
                        (
                            cv2.cvtColor(np.array(m).astype('uint8'), cv2.COLOR_GRAY2RGB)
                            * COLOR[cl]
                        ).astype(
                            np.uint8,
                        ),
                        0.5,
                        0,
                    )
                    img_g_cl = cv2.addWeighted(
                        np.array(img_0.copy()).astype('uint8'),
                        1,
                        (
                            cv2.cvtColor(np.array(m).astype('uint8'), cv2.COLOR_GRAY2RGB)
                            * COLOR[cl]
                        ).astype(
                            np.uint8,
                        ),
                        0.5,
                        0,
                    )
                    # Prediction
                    img_p = cv2.addWeighted(
                        np.array(img_p).astype('uint8'),
                        1,
                        (
                            cv2.cvtColor(np.array(m_p).astype('uint8'), cv2.COLOR_GRAY2RGB)
                            * COLOR[cl]
                        ).astype(
                            np.uint8,
                        ),
                        0.5,
                        0,
                    )
                    img_p_cl = cv2.addWeighted(
                        np.array(img_0.copy()).astype('uint8'),
                        1,
                        (
                            cv2.cvtColor(np.array(m_p).astype('uint8'), cv2.COLOR_GRAY2RGB)
                            * COLOR[cl]
                        ).astype(
                            np.uint8,
                        ),
                        0.5,
                        0,
                    )
                    res = np.hstack((img_0, img_g_cl))
                    res = np.hstack((res, img_p_cl))
                    logger.report_image(cl, f'Experiment {idy}', iteration=EPOCH, image=res)
                res = np.hstack((img_0, img_g))
                res = np.hstack((res, img_p))
                logger.report_image('All class', f'Experiment {idy}', iteration=EPOCH, image=res)

            histogram = []
            for num, cl in enumerate(self.classes):
                histogram.append(iou[:, num].mean().cpu().numpy())

            Logger.current_logger().report_histogram(
                'classes_metrics',
                'Validation',
                iteration=EPOCH,
                values=histogram,
                xlabels=self.classes,
                xaxis='Classes',
                yaxis='IOU',
            )
            EPOCH += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='train',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    model_dir = 'models/model_7'  # TODO: Is it a temporary solution?
    task = Task.init(  # TODO: this variable value is not used. Is it fine?
        project_name=cfg.project_name,
        task_name=cfg.task_name,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    dataset_path = ds.get(
        dataset_name=cfg.dataset_name,
        dataset_project=cfg.project_name,
    ).get_local_copy()

    train_dataset = OCTDataset(
        input_size=cfg.input_size,
        mask_dir=f'{dataset_path}/train/mask',
        img_dir=f'{dataset_path}/train/img',
        classes=cfg.classes,
    )

    test_dataset = OCTDataset(
        input_size=cfg.input_size,
        mask_dir=f'{dataset_path}/val/mask',
        img_dir=f'{dataset_path}/val/img',
        classes=cfg.classes,
    )

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

    model = OCTSegmentationModel(
        cfg.architecture,
        cfg.encoder,
        in_channels=3,
        classes=cfg.classes,
    )

    checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor='test/loss',
        mode='min',
        dirpath=f'{model_dir}/ckpt/',
        filename='sample-mnist-{epoch:02d}-{test/loss:.2f}',  # TODO: change name
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/')
    trainer = pl.Trainer(
        devices=-1,
        accelerator=cfg.device,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            checkpoint,
        ],
        enable_checkpointing=True,
        log_every_n_steps=cfg.batch_size,
        default_root_dir=model_dir,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )


if __name__ == '__main__':
    main()
