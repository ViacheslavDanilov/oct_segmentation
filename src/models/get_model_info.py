import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

# from src.models.smp.model import OCTSegmentationModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='get_model_info',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # model = OCTSegmentationModel(
    #     arch='Unet',
    #     encoder_name='se_resnet50',
    #     model_name='test_model',
    #     in_channels=3,
    #     classes=[
    #         'Arteriole lumen',
    #         'Arteriole media',
    #         'Arteriole adventitia',
    #         'Capillary lumen',
    #         'Capillary wall',
    #         'Venule lumen',
    #         'Venule wall',
    #         'Immune cells',
    #         'Nerve trunks',
    #     ],
    #     lr=0.0001,
    #     optimizer_name='RAdam',
    #     save_img_per_epoch=False,
    # )
    # model.eval()
    #
    # macs, params = get_model_complexity_info(
    #     model,
    #     (3, 896, 896),
    #     as_strings=True,
    #     print_per_layer_stat=True,
    #     verbose=True,
    # )
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
