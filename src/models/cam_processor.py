from typing import List

import numpy as np
import torch
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMElementWise,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    XGradCAM,
)
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.utils.image import show_cam_on_image


class CAMProcessor:
    """Class for processing images with Class Activation Mapping (CAM) methods.

    Attributes:
        CAM_METHODS (dict): Dictionary mapping CAM method names to their corresponding classes.
        model: The model used for CAM processing.
        device (str): The device on which the model and processing will be executed.
        cam_method (str): The name of the CAM method to use.
        target_layers (List): List of target layers for CAM processing.

    Methods:
        __init__: Initializes a CAMProcessor instance.
        process_image: Processes an image with the specified CAM method.
        _get_cam_method: Internal method to retrieve the CAM class based on the method name.
    """

    CAM_METHODS = {
        'GradCAM': GradCAM,
        'HiResCAM': HiResCAM,
        'GradCAMElementWise': GradCAMElementWise,
        'GradCAMPlusPlus': GradCAMPlusPlus,
        'XGradCAM': XGradCAM,
        'AblationCAM': AblationCAM,
        'EigenCAM': EigenCAM,
        'EigenGradCAM': EigenGradCAM,
        'LayerCAM': LayerCAM,
    }

    def __init__(
        self,
        model,
        device: str = 'cpu',
        cam_method: str = 'GradCAM',
        target_layers: List = None,
        percentile: int = 75,
    ) -> None:
        self.model = model
        self.cam_method = self._get_cam_method(cam_method)
        self.cam_metric_road = ROADMostRelevantFirst(percentile=percentile)
        self.cam_metric_conf = CamMultImageConfidenceChange()
        self.device = device
        self.target_layers = target_layers

    def _get_cam_method(self, cam_method):
        if cam_method not in self.CAM_METHODS:
            raise ValueError(f'Invalid CAM method: {cam_method}')
        return self.CAM_METHODS[cam_method]

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        image = (image / 255).astype('float32')
        image = image.transpose([2, 0, 1]).astype('float32')
        input_tensor = torch.Tensor(image).to(self.device)
        return input_tensor

    @staticmethod
    def _deprocess_image(
        image: torch.Tensor,
    ) -> np.ndarray:
        image = image.detach().cpu().numpy()
        image = image.squeeze().transpose([1, 2, 0])
        image = (image * 255).astype('uint8')
        return image

    def extract_activation_map(
        self,
        image: np.ndarray,
        class_idx: int,
        class_mask: np.ndarray,
    ):
        input_tensor = self._preprocess_image(image)
        targets = [SemanticSegmentationTarget(class_idx, class_mask)]
        with self.cam_method(model=self.model, target_layers=self.target_layers) as cam:
            mask_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        return mask_cam

    def compute_metrics(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_idx: int,
        class_mask: np.ndarray,
    ) -> np.ndarray:
        targets = [SemanticSegmentationTarget(class_idx, class_mask)]
        input_tensor = self._preprocess_image(image)
        score_road, vis = self.cam_metric_road(
            input_tensor=input_tensor,
            cams=np.array([mask]),
            targets=targets,
            model=self.model,
            return_visualization=True,
        )
        vis_out = self._deprocess_image(vis)
        # # TODO: visualization?
        # vis = vis.cpu().detach()
        # vis = vis.permute(0, 2, 3, 1).numpy().round()
        # vis = vis[0]

        score_conf, vis = self.cam_metric_conf(
            input_tensor=input_tensor,
            cams=np.array([1 - mask]),
            targets=targets,
            model=self.model,
            return_visualization=True,
        )

        return score_road[0], score_conf[0]

    @staticmethod
    def overlay_activation_map(
        image: np.ndarray,
        mask: np.ndarray,
        image_weight: float = 0.5,
    ) -> np.ndarray:
        img = (image / 255).astype('float32')
        fused_img = show_cam_on_image(
            img=img,
            mask=mask,
            use_rgb=False,
            image_weight=image_weight,
        )
        return fused_img


class SemanticSegmentationTarget:
    """Represents a semantic segmentation target.

    Attributes:
        category (int): The category of the target.
        mask (torch.Tensor): The mask associated with the target.

    Methods:
        __init__: Initializes a SemanticSegmentationTarget instance.
        __call__: Computes the target based on the model output.
    """

    def __init__(
        self,
        category: int,
        mask: np.ndarray,
    ) -> None:
        self.category = category
        if torch.cuda.is_available():
            self.mask = torch.from_numpy(mask).cuda()
        else:
            self.mask = torch.from_numpy(mask)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()
