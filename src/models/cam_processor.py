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
        _preprocess_image: Preprocesses the input image.
        get_targets: Gets the target for CAM processing.
        extract_activation_map: Extracts the activation map for the input image.
        overlay_activation_map: Overlays the activation map on the input image.
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
    ) -> None:
        self.model = model
        self.cam_method = self._get_cam_method(cam_method)
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
        image = image.transpose([2, 0, 1]).astype('float32')
        input_tensor = torch.Tensor(image).to(self.device)
        return input_tensor

    @staticmethod
    def get_targets(
        class_idx: int,
        class_mask: np.ndarray,
    ):
        return [SemanticSegmentationTarget(class_idx, class_mask)]

    def extract_activation_map(
        self,
        image: np.ndarray,
        targets: List,
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
    ):
        input_tensor = self._preprocess_image(image)
        with self.cam_method(model=self.model, target_layers=self.target_layers) as cam:
            mask_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )[0, :]
        return mask_cam

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
