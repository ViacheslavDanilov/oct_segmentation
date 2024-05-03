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
    ) -> None:
        self.model = model
        self.cam_method = self._get_cam_method(cam_method)
        self.device = device
        self.target_layers = target_layers

    def _get_cam_method(self, cam_method):
        if cam_method not in self.CAM_METHODS:
            raise ValueError(f'Invalid CAM method: {cam_method}')
        return self.CAM_METHODS[cam_method]

    def process_image(
        self,
        img: np.ndarray,
        class_idx: int,
        class_mask: np.ndarray,
    ):
        input_tensor = torch.Tensor(np.array(img)).to(self.device)
        targets = [SemanticSegmentationTarget(class_idx, class_mask)]
        with self.cam_method(model=self.model, target_layers=self.target_layers) as cam:
            img_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        return img_cam


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
        self.mask = (
            torch.from_numpy(mask).cuda() if torch.cuda.is_available() else torch.from_numpy(mask)
        )

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()
