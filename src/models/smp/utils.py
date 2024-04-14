import os
from csv import DictWriter
from typing import List, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torchvision
from PIL import Image

import wandb


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


def get_metrics(
    mask,
    pred_mask,
    loss,
    classes,
):
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_mask.long(),
        mask.long(),
        mode='multilabel',
        num_classes=len(classes),
    )
    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    dice = 2 * iou.cpu().numpy() / (iou.cpu().numpy() + 1)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    precision = smp.metrics.precision(tp, fp, fn, tn)
    recall = smp.metrics.sensitivity(tp, fp, fn, tn)
    return {
        'loss': loss.detach().cpu().numpy(),
        'iou': iou.cpu().numpy(),
        'dice': dice,
        'recall': recall.cpu().numpy(),
        'precision': precision.cpu().numpy(),
        'f1': f1.cpu().numpy(),
    }


def save_metrics_on_epoch(
    metrics_epoch: List[dict],
    split: str,
    model_name: str,
    classes: List[str],
    epoch: int,
    log_dict,
) -> None:
    header_w = False
    if not os.path.exists(f'models/{model_name}/metrics.csv'):
        header_w = True

    metrics_name = metrics_epoch[0].keys()
    metrics = {}
    for metric_name in metrics_name:
        for batch in metrics_epoch:
            if metric_name not in metrics:
                metrics[metric_name] = (
                    batch[metric_name]
                    if batch[metric_name].size == 1
                    else np.mean(
                        batch[metric_name],
                        axis=0,
                    )
                )
            else:
                if batch[metric_name].size == 1:
                    metrics[metric_name] = np.mean((batch[metric_name], metrics[metric_name]))
                else:
                    metrics[metric_name] = np.mean(
                        (np.mean(batch[metric_name], axis=0), metrics[metric_name]),
                        axis=0,
                    )

    metrics_log = {
        f'{split}/loss_mean': metrics['loss'],
        f'{split}/iou_mean': metrics['iou'].mean(),
        f'{split}/dice_mean': metrics['dice'].mean(),
        f'{split}/precision_mean': metrics['precision'].mean(),
        f'{split}/recall_mean': metrics['recall'].mean(),
        f'{split}/f1_mean': metrics['f1'].mean(),
    }

    metrics_l = metrics_log.copy()
    metrics_l['epoch'] = epoch
    wandb.log(metrics_l, step=epoch)  # type: ignore

    with open(f'models/{model_name}/metrics.csv', 'a', newline='') as f_object:
        fieldnames = [
            'Epoch',
            'Loss',
            'IoU',
            'Dice',
            'Precision',
            'Recall',
            'F1',
            'Split',
            'Class',
        ]
        writer = DictWriter(f_object, fieldnames=fieldnames)
        if header_w:
            writer.writeheader()

        for num, cl in enumerate(classes):
            for metric_name in [
                'iou',
                'dice',
                'precision',
                'recall',
                'f1',
            ]:
                metrics_log[f'{split}/{metric_name} ({cl})'] = metrics[metric_name][num]
                metrics_log[f'{metric_name} {split}/{cl}'] = metrics[metric_name][num]
            writer.writerow(
                {
                    'Epoch': epoch,
                    'Loss': metrics['loss'],
                    'IoU': metrics['iou'][num],
                    'Dice': metrics['dice'][num],
                    'Precision': metrics['precision'][num],
                    'Recall': metrics['recall'][num],
                    'F1': metrics['f1'][num],
                    'Split': split,
                    'Class': cl,
                },
            )
        writer.writerow(
            {
                'Epoch': epoch,
                'Loss': metrics['loss'],
                'IoU': metrics['iou'].mean(),
                'Dice': metrics['dice'].mean(),
                'Precision': metrics['precision'].mean(),
                'Recall': metrics['recall'].mean(),
                'F1': metrics['f1'].mean(),
                'Split': split,
                'Class': 'Mean',
            },
        )
        log_dict(metrics_log, on_epoch=True)
        f_object.close()


def get_img_mask_union_pil(
    img: Image,
    mask: np.ndarray,
    color: tuple[int],
    alpha: float = 0.85,
):
    mask *= alpha
    mask *= 255
    class_img = Image.new('RGB', size=img.size, color=color)
    img.paste(class_img, (0, 0), Image.fromarray(mask.astype('uint8')))
    return img


def calculate_iou(gt_mask, pred_mask):
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    overlap = pred_mask * gt_mask
    union = (pred_mask + gt_mask) > 0
    iou = overlap.sum() / float(union.sum())
    return iou


def get_img_color_mask(
    img_0: np.ndarray,
    alpha_0: float,
    img_1: np.ndarray,
    alpha_1: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2BGR) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


def to_tensor(
    x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


get_tensor = torchvision.transforms.ToTensor()


def preprocessing_img(
    img_path: str,
    input_size: int,
):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (input_size, input_size))
    image = to_tensor(np.array(image))
    return image
