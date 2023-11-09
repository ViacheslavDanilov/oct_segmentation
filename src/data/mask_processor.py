import cv2
import numpy as np


class MaskProcessor:
    """MaskProcessor is a class for processing binary masks.

    It provides methods for smoothing, and artifact removal of masks.
    """

    @staticmethod
    def smooth_mask(
        mask: np.ndarray,
    ) -> np.ndarray:
        min_dim = min(mask.shape)
        kernel_size = max(int(0.01 * min_dim), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
        mask_fill = cv2.morphologyEx(mask_close, cv2.MORPH_DILATE, kernel)

        return mask_fill

    @staticmethod
    def remove_artifacts(
        mask: np.ndarray,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        sorted_areas = sorted(areas, reverse=True)[:3]
        biggest_contours = [cnt for cnt, area in zip(contours, areas) if area in sorted_areas]
        mask_new = np.zeros_like(mask)
        mask_new = cv2.drawContours(mask_new, biggest_contours, -1, 1, thickness=cv2.FILLED)

        return mask_new
