import datetime
import logging
import os

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models.smp.dataset import OCTDataModule
from src.models.smp.model import OCTSegmentationModel

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

    # Initialize ClearML task
    Task.init(
        project_name=cfg.project_name,
        task_name=cfg.task_name,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    # Initialize data module
    oct_data_module = OCTDataModule(
        dataset_name=cfg.dataset_name,
        project_name=cfg.project_name,
        input_size=cfg.input_size,
        classes=cfg.classes,
    )
    oct_data_module.prepare_data()
    oct_data_module.setup(stage='train')
    train_dataloader = oct_data_module.train_dataloader(
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        shuffle=True,
    )
    val_dataloader = oct_data_module.val_dataloader(
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    # Initialize callbacks
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor='test/loss',
        mode='min',
        dirpath=f'{model_dir}/ckpt/',
        filename='models_{epoch:02d}',
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
    )

    # Initialize model
    model = OCTSegmentationModel(
        cfg.architecture,
        cfg.encoder,
        in_channels=3,
        classes=cfg.classes,
        colors=cfg.classes_color,
    )

    # Initialize and tun trainer
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
        val_dataloaders=val_dataloader,
        # datamodule=oct_data_module    # TODO: you may directly pass DataModule here
        #                                       https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
    )


if __name__ == '__main__':
    main()
