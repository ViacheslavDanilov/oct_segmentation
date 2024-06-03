import gc
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
from src.models.smp.dataset import OCTDataModule
from src.models.smp.model import OCTSegmentationModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='tune',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    os.environ['WANDB_API_KEY'] = '0a94ef68f2a7a8b709671d6ef76e61580d20da7f'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': f'{cfg.metric_type}/{cfg.metric_name}', 'goal': cfg.metric_sign},
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 2,
            'min_iter': 25,
            'max_iter': cfg.epochs,
        },
        'parameters': {
            'classes': {'value': list(cfg.classes)},
            'batch_size': {'value': cfg.batch_size},
            'epochs': {'value': cfg.epochs},
            'cuda_num': {'value': list(cfg.cuda_num)},
            # Variable hyperparameters
            'input_size': {
                'values': list(
                    range(cfg.input_size_min, cfg.input_size_max + 1, cfg.input_size_step),
                ),
            },
            'optimizer': {'values': list(cfg.optimizer)},
            'lr': {'values': list(cfg.learning_rate)},
            'architecture': {'values': list(cfg.architecture)},
            'encoder': {'values': list(cfg.encoder)},
            'data_dir': {'value': cfg.data_dir},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity='vladislavlaptev', project=cfg.project_name)
    wandb.agent(sweep_id=sweep_id, function=tune, count=350)

    # If the tuning is interrupted, use a specific sweep_id to keep tuning on the next call
    # wandb.agent(sweep_id='3t2kelpq', function=tune, count=200, entity='vladislavlaptev', project=cfg.project_name)

    print('\n\033[92m' + '-' * 100 + '\033[0m')
    print('\033[92m' + 'Tuning has finished!' + '\033[0m')
    print('\033[92m' + '-' * 100 + '\033[0m')


def tune(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name
        print('\033[92m' + '\n********** Run: {:s} **********\n'.format(run_name) + '\033[0m')
        model_dir = os.path.join('models', f'{run_name}')
        callbacks = [
            LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=False,
            ),
        ]

        os.makedirs(f'{model_dir}')
        oct_data_module = OCTDataModule(
            data_dir=config.data_dir,
            classes=config.classes,
            input_size=config.input_size,
            batch_size=config.batch_size,
            num_workers=os.cpu_count(),
            use_augmentation=True,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir='logs/',
        )

        # Initialize model
        model = OCTSegmentationModel(
            arch=config.architecture,
            encoder_name=config.encoder,
            optimizer_name=config.optimizer,
            in_channels=3,
            classes=config.classes,
            model_name=run_name,
            lr=config.lr,
            img_save_interval=None,
        )

        # Initialize and run trainer
        trainer = pl.Trainer(
            devices=config.cuda_num,
            accelerator='cuda',
            max_epochs=config.epochs,
            logger=tb_logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            log_every_n_steps=config.batch_size,
            default_root_dir=model_dir,
        )

        try:
            trainer.fit(
                model,
                datamodule=oct_data_module,
            )
        except Exception:
            print('Run status: CUDA out-of-memory error or HyperBand stop')
        else:
            print('Run status: Success')
        finally:
            print('Reset memory and clean garbage')
            gc.collect()
            torch.cuda.empty_cache()
        wandb.join()


if __name__ == '__main__':
    main()
