import datetime
import json
import logging
import os
import ssl

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import PROJECT_DIR
from src.models.smp.dataset import OCTDataModule
from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import pick_device

ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

os.environ['WANDB_API_KEY'] = '0a94ef68f2a7a8b709671d6ef76e61580d20da7f'


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='train',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = str(os.path.join(PROJECT_DIR, cfg.data_dir))
    save_dir = str(os.path.join(PROJECT_DIR, cfg.save_dir))

    today = datetime.datetime.today()
    task_name = f'{cfg.architecture}_{cfg.encoder}_{today.strftime("%d%m_%H%M")}'
    model_dir = f'{save_dir}/{task_name}'

    device = pick_device(cfg.device)

    hyperparams = {
        'architecture': cfg.architecture,
        'encoder': cfg.encoder,
        'input_size': cfg.input_size,
        'classes': list(cfg.classes),
        'num_classes': len(cfg.classes),
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'use_augmentation': cfg.use_augmentation,
        'epochs': cfg.epochs,
        'device': device,
        'data_dir': data_dir,
    }

    wandb.init(config=hyperparams, project='oct_segmentation', name=task_name)  # type: ignore

    callbacks = [
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False,
        ),
    ]
    if cfg.img_save_interval is not None:
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
        data_dir=data_dir,
        classes=cfg.classes,
        input_size=hyperparams['input_size'],
        batch_size=hyperparams['batch_size'],
        num_workers=os.cpu_count(),
        use_augmentation=cfg.use_augmentation,
    )

    # Initialize model
    model = OCTSegmentationModel(
        arch=hyperparams['architecture'],
        encoder_name=hyperparams['encoder'],
        optimizer_name=hyperparams['optimizer'],
        input_size=hyperparams['input_size'],
        in_channels=3,
        classes=cfg.classes,
        model_name=task_name,
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        img_save_interval=cfg.img_save_interval,
        save_wandb_media=cfg.save_wandb_media,
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
            indent=2,
        )

    # Initialize and tun trainer
    trainer = pl.Trainer(
        devices='auto',
        accelerator=device,
        max_epochs=hyperparams['epochs'],
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=hyperparams['batch_size'],
        default_root_dir=model_dir,
    )
    trainer.fit(
        model,
        datamodule=oct_data_module,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()
