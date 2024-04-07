from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from src.models.smp.utils import get_metrics, log_predict_model_on_epoch, save_metrics_on_epoch


class OCTSegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of OCT images."""

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        model_name: str,
        in_channels: int,
        classes: List[str],
        lr: float = 0.0001,
        optimizer_name: str = 'Adam',
        save_img_per_epoch: int = None,
        **kwargs,
    ):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=len(classes),
            **kwargs,
        )

        self.classes = classes
        self.epoch = -1
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        self.training_step_outputs = []  # type: ignore
        self.validation_step_outputs = []  # type: ignore
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.model_name = model_name
        self.lr = lr
        self.optimizer = optimizer_name
        self.save_img_per_epoch = save_img_per_epoch

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def training_step(
        self,
        batch: List[float],
        batch_idx: int,
    ):
        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()  # type: ignore
        pred_mask = (prob_mask > 0.5).float()

        self.log('training/loss', loss, prog_bar=True, on_epoch=True)
        self.training_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
                classes=self.classes,
            ),
        )
        return {
            'loss': loss,
        }

    def on_train_epoch_end(self):
        save_metrics_on_epoch(
            metrics_epoch=self.training_step_outputs,
            split='train',
            model_name=self.model_name,
            classes=self.classes,
            epoch=self.epoch - 1,
            log_dict=self.log_dict,
        )
        self.training_step_outputs.clear()

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        img, mask = batch
        logits_mask = self.forward(img)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
                classes=self.classes,
            ),
        )
        self.log(
            'val/f1',
            np.mean(self.validation_step_outputs[-1]['F1']).mean(),
            prog_bar=True,
            on_epoch=True,
        )
        if self.save_img_per_epoch is not None:
            if batch_idx == 0 and self.epoch % self.save_img_per_epoch == 0:
                log_predict_model_on_epoch(
                    img=img,
                    mask=mask,
                    pred_mask=pred_mask,
                    classes=self.classes,
                    epoch=self.epoch,
                    model_name=self.model_name,
                )

    def on_validation_epoch_end(self):
        if self.epoch > 0:
            save_metrics_on_epoch(
                metrics_epoch=self.validation_step_outputs,
                split='test',
                model_name=self.model_name,
                classes=self.classes,
                epoch=self.epoch,
                log_dict=self.log_dict,
            )
        self.validation_step_outputs.clear()
        self.epoch += 1

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RAdam':
            return torch.optim.RAdam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SAdam':
            return torch.optim.SparseAdam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

    def predict(
        self,
        images: np.ndarray,
        device: str,
    ):
        y_hat = self.model(torch.Tensor(images).to(device)).cpu().detach()
        masks = y_hat.sigmoid()
        masks = (masks > 0.5).float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.numpy().round()
        return masks
