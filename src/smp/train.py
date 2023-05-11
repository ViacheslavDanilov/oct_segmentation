import datetime
import logging
import os

import hydra
import pytorch_lightning as pl
from clearml import Dataset as ds
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from src.smp.dataset import OCTDataset, get_img_augmentation
from src.smp.model import OCTSegmentationModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='train_smp_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    today = datetime.datetime.today()
    model_dir = f'models/{cfg.project_name}/{cfg.task_name}_#{today.strftime("%m-%d-%H.%M")}'
    Task.init(
        project_name=cfg.project_name,
        task_name=cfg.task_name,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    dataset_path = ds.get(
        dataset_name=cfg.dataset_name,
        dataset_project=cfg.project_name,
    ).get_local_copy()

    # TODO: replace train_dataset/train_dataloader and test_dataset/test_dataloader with data_module
    train_dataset = OCTDataset(
        input_size=cfg.input_size,
        mask_dir=f'{dataset_path}/train/ann',
        img_dir=f'{dataset_path}/train/img',
        classes=cfg.classes,
        augmentation=get_img_augmentation(cfg.input_size),
    )

    test_dataset = OCTDataset(
        input_size=cfg.input_size,
        mask_dir=f'{dataset_path}/val/ann',
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
        colors=cfg.classes_color,
    )

    checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor='test/loss',
        mode='min',
        dirpath=f'{model_dir}/ckpt/',
        filename='models_{epoch:02d}',
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

    # TODO: replace with OCTDataModule
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )


if __name__ == '__main__':
    main()
