import datetime
import json
import logging
import os
import ssl

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import PROJECT_DIR
from src.models.smp.dataset import OCTDataModule
from src.models.smp.model import OCTSegmentationModel

ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='train_smp',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = os.path.join(PROJECT_DIR, cfg.data_dir)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)

    today = datetime.datetime.today()
    task_name = f'{cfg.architecture}_{cfg.encoder}_{today.strftime("%d%m_%H%M")}'
    model_dir = os.path.join(save_dir, f'{task_name}')

    hyperparameters = {
        'architecture': cfg.architecture,
        'encoder': cfg.encoder,
        'input_size': cfg.input_size,
        'classes': list(cfg.classes),
        'num_classes': len(cfg.classes),
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer,
        'lr': cfg.lr,
        'epochs': cfg.epochs,
        'device': cfg.device,
        'data_dir': data_dir,
    }

    wandb.init(
        config=hyperparameters,
        project='oct_segmentation',
        name=task_name,
    )

    callbacks = [
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False,
        ),
    ]
    if cfg.log_artifacts:
        os.makedirs(f'{model_dir}/images_per_epoch', exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                save_top_k=1,
                monitor='val/loss',
                mode='min',
                dirpath=f'{model_dir}/',
                filename='weights',
            ),
        )
    else:
        os.makedirs(f'{model_dir}', exist_ok=True)

    # Initialize data module
    oct_data_module = OCTDataModule(
        input_size=hyperparameters['input_size'],
        classes=cfg.classes,
        batch_size=hyperparameters['batch_size'],
        num_workers=os.cpu_count(),
        data_dir=data_dir,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
    )

    # Initialize model
    model = OCTSegmentationModel(
        arch=hyperparameters['architecture'],
        encoder_name=hyperparameters['encoder'],
        optimizer_name=hyperparameters['optimizer'],
        in_channels=3,
        classes=cfg.classes,
        model_name=task_name,
        lr=hyperparameters['lr'],
        save_img_per_epoch=5 if cfg.log_artifacts else None,
    )
    with open(f'{model_dir}/config.json', 'w') as file:
        json.dump(
            {
                'model_name': f'{cfg.architecture}_{cfg.encoder}',
                'architecture': cfg.architecture,
                'encoder': cfg.encoder,
                'input_size': cfg.input_size,
                'classes': list(cfg.classes),
                'batch_size': cfg.batch_size,
                'optimizer': cfg.optimizer,
                'lr': cfg.lr,
            },
            file,
        )

    # Initialize and tun trainer
    trainer = pl.Trainer(
        devices=cfg.cuda_num,
        accelerator=cfg.device,
        max_epochs=hyperparameters['epochs'],
        logger=tb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=hyperparameters['batch_size'],
        default_root_dir=model_dir,
    )
    trainer.fit(
        model,
        datamodule=oct_data_module,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()
