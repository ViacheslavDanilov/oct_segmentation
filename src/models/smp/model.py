from typing import List, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from clearml import Logger


def get_img_mask_union(
    img_0: np.ndarray,
    alpha_0: float,
    img_1: np.ndarray,
    alpha_1: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2RGB) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


class OCTSegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of OCT images."""

    def __init__(
        self,
        architecture: str,
        encoder_name: str,
        optimizer: str,
        lr: float,
        in_channels: int,
        classes: List[str],
        colors: List[Tuple[int, int, int]],
        **kwargs,
    ):
        super().__init__()
        self.model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=len(classes),
            **kwargs,
        )

        self.optimizer = optimizer
        self.lr = lr
        self.classes = classes
        self.colors = colors
        self.epoch = -1
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.training_histogram = np.zeros(len(self.classes))
        self.training_histogram_best_mean = np.zeros(len(self.classes))
        self.validation_histogram = np.zeros(len(self.classes))
        self.validation_histogram_best_mean = np.zeros(len(self.classes))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

        self.my_logger = Logger.current_logger()

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    # TODO: ClearML logging should be a separate method or class.
    def training_step(
        self,
        batch,
        batch_idx: int,
    ):
        if batch_idx == 0:
            if self.epoch > 0:
                self.my_logger.report_histogram(
                    'Last metrics',
                    'Training',
                    iteration=self.epoch,
                    values=self.training_histogram,
                    xlabels=self.classes,
                    xaxis='Classes',
                    yaxis='IOU',
                )
                if np.mean(self.training_histogram) > np.mean(self.training_histogram_best_mean):
                    self.training_histogram_best_mean = self.training_histogram
                    self.my_logger.report_histogram(
                        'Best metrics',
                        'Training',
                        iteration=self.epoch,
                        values=self.training_histogram_best_mean,
                        xlabels=self.classes,
                        xaxis='Classes',
                        yaxis='IOU',
                    )
                self.training_histogram = np.zeros(len(self.classes))

        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode='multilabel',
            num_classes=len(self.classes),
        )

        # TODO: Track all metrics
        smp.metrics.accuracy(tp, fp, fn, tn)
        smp.metrics.f1_score(tp, fp, fn, tn)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn)
        smp.metrics.positive_predictive_value(tp, fp, fn, tn)
        smp.metrics.sensitivity(tp, fp, fn, tn)
        smp.metrics.sensitivity(tp, fp, fn, tn)
        smp.metrics.specificity(tp, fp, fn, tn)

        self.log('train/loss', loss, prog_bar=True, on_epoch=True)

        for num, cl in enumerate(self.classes):
            self.training_histogram[num] += iou_score[:, num].mean().cpu().numpy()
            if batch_idx != 0:
                self.training_histogram[num] /= 2

        metrics = {
            'train/IOU (mean)': iou_score.mean(),
        }
        for num, cl in enumerate(self.classes):
            metrics[f'train/IOU ({cl})'] = iou_score[:, num].mean()
        self.log_dict(metrics, on_epoch=True)

        return {
            'loss': loss,
        }

    # TODO: ClearML logging should be a separate method or class.
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
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode='multilabel',
            num_classes=len(self.classes),
        )

        # TODO: Track all metrics
        smp.metrics.accuracy(tp, fp, fn, tn)
        smp.metrics.f1_score(tp, fp, fn, tn)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn)
        smp.metrics.positive_predictive_value(tp, fp, fn, tn)
        smp.metrics.sensitivity(tp, fp, fn, tn)
        smp.metrics.sensitivity(tp, fp, fn, tn)
        smp.metrics.specificity(tp, fp, fn, tn)

        metrics = {
            'test/IOU (mean)': iou_score.mean(),
        }
        for num, cl in enumerate(self.classes):
            metrics[f'test/IOU ({cl})'] = iou_score[:, num].mean()
        self.log_dict(metrics, on_epoch=True)
        self.log('test/loss', loss, prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            if self.epoch > 0:
                self.my_logger.report_histogram(
                    'Last metrics',
                    'Test',
                    iteration=self.epoch,
                    values=self.validation_histogram,
                    xlabels=self.classes,
                    xaxis='Classes',
                    yaxis='IOU',
                )
                if np.mean(self.validation_histogram) > np.mean(
                    self.validation_histogram_best_mean,
                ):
                    self.validation_histogram_best_mean = self.validation_histogram
                    self.my_logger.report_histogram(
                        'Best metrics',
                        'Test',
                        iteration=self.epoch,
                        values=self.validation_histogram_best_mean,
                        xlabels=self.classes,
                        xaxis='Classes',
                        yaxis='IOU',
                    )
                self.validation_histogram = np.zeros(len(self.classes))

            self.epoch += 1

            img = img.permute(0, 2, 3, 1)
            img = img.squeeze().cpu().numpy().round()
            mask = mask.squeeze().cpu().numpy().round()
            pred_mask = pred_mask.squeeze().cpu().numpy().round()
            for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
                img_ = np.array(img_)
                img_g = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_p = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_0 = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                for cl, m, m_p in zip(self.classes, mask_, pr_mask):
                    # Groundtruth
                    img_g = get_img_mask_union(
                        img_0=img_g,
                        alpha_0=1,
                        img_1=m,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    img_g_cl = get_img_mask_union(
                        img_0=img_0.copy(),
                        alpha_0=1,
                        img_1=m,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    # Prediction
                    img_p = get_img_mask_union(
                        img_0=img_p,
                        alpha_0=1,
                        img_1=m_p,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    img_p_cl = get_img_mask_union(
                        img_0=img_0.copy(),
                        alpha_0=1,
                        img_1=m_p,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    res = np.hstack((img_0, img_g_cl))
                    res = np.hstack((res, img_p_cl))
                    self.my_logger.report_image(
                        cl,
                        f'Experiment {idy}',
                        image=res,
                        iteration=self.epoch,
                    )
                res = np.hstack((img_0, img_g))
                res = np.hstack((res, img_p))
                self.my_logger.report_image(
                    'All class',
                    f'Experiment {idy}',
                    image=res,
                    iteration=self.epoch,
                )

        for num, cl in enumerate(self.classes):
            self.validation_histogram[num] += iou_score[:, num].mean().cpu().numpy()
            if batch_idx != 0:
                self.validation_histogram[num] /= 2

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RAdam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')
