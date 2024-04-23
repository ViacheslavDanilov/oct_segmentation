import logging
import os

import hydra
import segmentation_models_pytorch as smp
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info

from src import PROJECT_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='get_model_info',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    model = smp.create_model(
        arch=cfg.model_name,
        encoder_name=cfg.encoder_name,
    )
    if cfg.evaluation_mode:
        model.eval()

    flops, params = get_model_complexity_info(
        model=model,
        input_res=tuple(cfg.input_size[::-1]),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=False,
        output_precision=1,
    )
    log.info(f'Number of parameters: {params}')
    log.info(f'Computational complexity: {flops}')


if __name__ == '__main__':
    main()
