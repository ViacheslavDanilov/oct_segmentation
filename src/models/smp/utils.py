import os
from csv import DictWriter
from typing import List, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from PIL import Image


def get_metrics(
    mask,
    pred_mask,
    loss,
    eps: float = 1e-7,
):
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_mask.long(),
        mask.long(),
        mode='multilabel',
    )
    iou = smp.metrics.iou_score(tp, fp, fn, tn, zero_division=eps)
    dice = 2 * iou.cpu().numpy() / (iou.cpu().numpy() + 1)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, zero_division=eps)
    precision = smp.metrics.precision(tp, fp, fn, tn, zero_division=eps)
    recall = smp.metrics.sensitivity(tp, fp, fn, tn, zero_division=eps)
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
    best_metrics: dict = None,
) -> dict:
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
        f'{split}/loss': metrics['loss'],
        f'{split}/iou': metrics['iou'].mean(),
        f'{split}/dice': metrics['dice'].mean(),
        f'{split}/precision': metrics['precision'].mean(),
        f'{split}/recall': metrics['recall'].mean(),
        f'{split}/f1': metrics['f1'].mean(),
    }

    # best metrics
    if best_metrics is not None:
        for metric_name in ['iou', 'dice', 'precision', 'recall']:
            if metric_name not in best_metrics:
                best_metrics[metric_name] = {
                    'value': metrics_log[f'{split}/{metric_name}'],
                    'epoch': epoch,
                }
                wandb.run.summary[f'best_{metric_name}'] = metrics_log[f'{split}/{metric_name}']
                wandb.run.summary[f'best_{metric_name}_epoch'] = epoch
            else:
                if metrics_log[f'{split}/{metric_name}'] > best_metrics[metric_name]['value']:
                    best_metrics[metric_name] = {
                        'value': metrics_log[f'{split}/{metric_name}'],
                        'epoch': epoch,
                    }
                    wandb.run.summary[f'best_{metric_name}'] = metrics_log[f'{split}/{metric_name}']
                    wandb.run.summary[f'best_{metric_name}_epoch'] = epoch

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
                metrics_log[f'{split}/{metric_name} ({cl})'] = (
                    metrics[metric_name][num] if len(classes) > 1 else metrics[metric_name]
                )
                metrics_log[f'{metric_name} {split}/{cl}'] = (
                    metrics[metric_name][num] if len(classes) > 1 else metrics[metric_name]
                )
            writer.writerow(
                {
                    'Epoch': epoch,
                    'Loss': metrics['loss'],
                    'IoU': metrics['iou'][num] if len(classes) > 1 else metrics['iou'],
                    'Dice': metrics['dice'][num] if len(classes) > 1 else metrics['dice'],
                    'Precision': (
                        metrics['precision'][num] if len(classes) > 1 else metrics['precision']
                    ),
                    'Recall': metrics['recall'][num] if len(classes) > 1 else metrics['recall'],
                    'F1': metrics['f1'][num] if len(classes) > 1 else metrics['f1'],
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
        f_object.close()
    return best_metrics


def calculate_iou(gt_mask, pred_mask):
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    overlap = pred_mask * gt_mask
    union = (pred_mask + gt_mask) > 0
    iou = overlap.sum() / float(union.sum())
    return iou


def calculate_dice(pred_mask, gt_mask):
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    intersection = 2 * (gt_mask * pred_mask).sum()
    return 2 * intersection / (gt_mask.sum() + pred_mask.sum())


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


def preprocessing_img(
    img_path: str,
    input_size: int,
):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (input_size, input_size))
    image = to_tensor(image)
    return image


def pick_device(
    option: str,
) -> str:
    """Pick the appropriate device based on the provided option.

    Args:
        option (str): Available device option ('cpu', 'cuda', 'auto').

    Returns:
        str: Selected device.
    """
    if option == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif option in ['cpu', 'cuda']:
        return option
    else:
        raise ValueError("Invalid device option. Please specify 'cpu', 'cuda', or 'auto'.")
