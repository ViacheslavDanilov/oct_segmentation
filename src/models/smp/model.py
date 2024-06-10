from glob import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import tifffile
import torch
import wandb

from src.data.utils import CLASS_COLORS_BGR, CLASS_IDS, CLASS_IDS_REVERSED
from src.models.smp.utils import get_metrics, save_metrics_on_epoch


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
        weight_decay: float = 0.0001,
        optimizer_name: str = 'Adam',
        input_size: int = 512,
        img_save_interval: int | None = 1,
        save_wandb_media: bool = False,
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
        self.epoch = 0
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        self.training_step_outputs = []  # type: ignore
        self.validation_step_outputs = []  # type: ignore
        self.validation_best_metrics = {}  # type: ignore
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer_name
        self.input_size = input_size
        self.img_save_interval = img_save_interval
        self.save_wandb_media = save_wandb_media
        self.class_values = [CLASS_IDS[cl] for _, cl in enumerate(self.classes)]

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
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

        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        self.training_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
            ),
        )
        return {
            'loss': loss,
        }

    def on_train_epoch_end(self):
        _ = save_metrics_on_epoch(
            metrics_epoch=self.training_step_outputs,
            split='train',
            model_name=self.model_name,
            classes=self.classes,
            epoch=self.epoch,
        )
        self.training_step_outputs.clear()
        self.epoch += 1

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
            ),
        )
        self.log(
            'val/f1',
            float(np.mean(self.validation_step_outputs[-1]['f1']).mean()),
            prog_bar=True,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        if self.epoch > 0:
            self.validation_best_metrics = save_metrics_on_epoch(
                metrics_epoch=self.validation_step_outputs,
                split='test',
                model_name=self.model_name,
                classes=self.classes,
                epoch=self.epoch,
                best_metrics=self.validation_best_metrics,
            )
            if self.img_save_interval is not None and self.epoch % self.img_save_interval == 0:
                self.log_predict_model_on_epoch()
        else:
            self.epoch += 1
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == 'RAdam':
            return torch.optim.RAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == 'SAdam':
            return torch.optim.SparseAdam(
                self.parameters(),
                lr=self.lr,
            )
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

    def predict(
        self,
        images: np.ndarray,  # standard shape: (N, H, W, C)
        device: str,
    ) -> np.ndarray:
        # Convert images to tensor of shape (N, C, H, W) and move to device
        images_tensor = torch.Tensor(images.transpose((0, 3, 1, 2))).to(device)

        # Perform model prediction
        y_hat = self.model(images_tensor).cpu().detach()

        # Apply sigmoid and thresholding
        masks = (y_hat.sigmoid() > 0.5).float()

        # Convert masks back to numpy array
        masks = masks.permute(0, 2, 3, 1).numpy().round()

        return masks

    @staticmethod
    def to_tensor_shape(
        x: np.ndarray,
    ) -> np.ndarray:
        return x.transpose([2, 0, 1]).astype('float32')

    def log_predict_model_on_epoch(
        self,
    ):
        wandb_images = []
        for idx, img_path in enumerate(glob('data/visualization/img/*.png')):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_size, self.input_size))
            mask = tifffile.imread(f"{img_path.replace('img', 'mask').split('.')[0]}.tiff")
            mask = cv2.resize(
                mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            )

            pred_mask = self.predict(
                images=np.array([self.to_tensor_shape(img.copy())]),
                device='cuda',
            )[0]
            color_mask_gt = np.zeros(img.shape, dtype=np.uint8)
            color_mask_pred = np.zeros(img.shape, dtype=np.uint8)
            color_mask_pred[:, :] = (128, 128, 128)
            color_mask_gt[:, :] = (128, 128, 128)

            wandb_mask_inference = np.zeros((img.shape[0], img.shape[1]))
            wandb_mask_ground_truth = np.zeros((img.shape[0], img.shape[1]))
            for idy, cl in enumerate(self.classes):
                class_id = CLASS_IDS[cl] - 1
                color_mask_gt[mask[:, :, class_id] == 255] = CLASS_COLORS_BGR[cl]
                color_mask_pred[pred_mask[:, :, idy] == 1] = CLASS_COLORS_BGR[cl]
                wandb_mask_inference[pred_mask[:, :, idy] == 1] = CLASS_IDS[cl]
                wandb_mask_ground_truth[mask[:, :, class_id] == 255] = CLASS_IDS[cl]

            res = np.hstack((img, color_mask_gt))
            res = np.hstack((res, color_mask_pred))

            img_stem = Path(img_path).stem
            cv2.imwrite(
                f'models/{self.model_name}/images_per_epoch/{img_stem}_epoch_{str(self.epoch).zfill(3)}.png',
                res,
            )

            if self.save_wandb_media:
                wandb_images.append(
                    wandb.Image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        masks={
                            'predictions': {
                                'mask_data': wandb_mask_inference,
                                'class_labels': CLASS_IDS_REVERSED,
                            },
                            'ground_truth': {
                                'mask_data': wandb_mask_ground_truth,
                                'class_labels': CLASS_IDS_REVERSED,
                            },
                        },
                        caption=f'Example-{idx}',
                    ),
                )
        if self.save_wandb_media:
            wandb.log(
                {'Examples': wandb_images},
                step=self.epoch,
            )
