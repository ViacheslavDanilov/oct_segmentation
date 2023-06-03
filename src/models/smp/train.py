import logging
import os
import time

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import DataManager, OCTDataModule, OCTSegmentationModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='train_smp_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    timestamp = time.strftime('%d%m_%H%M%S', time.localtime())
    exp_name = f'{cfg.architecture}_{cfg.encoder}'
    model_dir = os.path.join('models', f'{cfg.project_name}', f'{exp_name}_{timestamp}')

    # Initialize ClearML task
    Task.init(
        project_name=cfg.project_name,
        task_name=exp_name,
        auto_connect_frameworks={
            'tensorboard': True,
            'pytorch': True,
        },
    )

    # Synchronize dataset with ClearML workspace
    data_manager = DataManager(
        data_dir=cfg.data_dir,
    )
    data_manager.prepare_data()

    # Initialize data module
    oct_data_module = OCTDataModule(
        input_size=cfg.input_size,
        classes=cfg.classes,
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
    )

    # Initialize callbacks
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor='test/loss',
        mode='min',
        dirpath=f'{model_dir}/ckpt/',
        filename='models_{epoch+1:03d}',
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
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
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
        datamodule=oct_data_module,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
