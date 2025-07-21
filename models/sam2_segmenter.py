"""
SAM2.1 segmenter for precise object segmentation.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import supervision as sv
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

class SAM2Segmenter:
    """SAM2.1 segmenter for precise object segmentation."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SAM2.1 segmenter.

        Args:
            device: Device to run inference on
        """
        self.device = device
        self.predictor = self._load_model()

    def _load_model(self) -> SAM2ImagePredictor:
        """Load SAM2.1 model."""
        try:
            # Load SAM2.1 model using the correct API with device specification
            predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=self.device)

            print(f"SAM2.1 model loaded successfully on {self.device}")
            return predictor

        except Exception as e:
            print(f"Error loading SAM2.1 model: {e}")
            print("Please ensure model files are downloaded correctly.")
            raise

    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.

        Args:
            image: Input image as numpy array (RGB format)
        """
        self.predictor.set_image(image)

    def segment_from_boxes(self, boxes: np.ndarray,
                          image: np.ndarray) -> List[np.ndarray]:
        """
        Segment objects from bounding boxes.

        Args:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            image: Input image as numpy array (RGB format)

        Returns:
            List of binary masks
        """
        self.set_image(image)

        masks = []
        for box in boxes:
            # Convert box to SAM2 format [x1, y1, x2, y2]
            input_box = np.array(box)

            # Get mask using SAM2.1 API with proper inference mode
            if self.device == "cuda":
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    mask, _, _ = self.predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
            else:
                with torch.inference_mode():
                    mask, _, _ = self.predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )

            # Convert torch tensor to numpy array and ensure boolean dtype
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            # Remove batch dimension if present
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask = mask.astype(bool)
            masks.append(mask)

        return masks

    def segment_from_points(self, points: np.ndarray,
                           labels: np.ndarray,
                           image: np.ndarray) -> List[np.ndarray]:
        """
        Segment objects from points.

        Args:
            points: Points in format [[x1, y1], [x2, y2], ...]
            labels: Point labels (1 for foreground, 0 for background)
            image: Input image as numpy array (RGB format)

        Returns:
            List of binary masks
        """
        self.set_image(image)

        # Get mask using SAM2.1 API with proper inference mode
        if self.device == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mask, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
        else:
            with torch.inference_mode():
                mask, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )

        # Convert torch tensor to numpy array and ensure boolean dtype
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        # Remove batch dimension if present
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask = mask.astype(bool)

        return [mask]

    def segment_combined(self, detections: sv.Detections,
                        image: np.ndarray) -> Dict[str, Any]:
        """
        Segment objects from detections.

        Args:
            detections: Supervision detections object
            image: Input image as numpy array (RGB format)

        Returns:
            Dictionary with segmentation results
        """
        if len(detections) == 0:
            return {
                'masks': [],
                'segmented_areas': [],
                'total_include_area': 0,
                'total_exclude_area': 0
            }

        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert normalized coordinates to pixel coordinates
        height, width = image.shape[:2]
        pixel_boxes = detections.xyxy.copy()

        # Check if boxes are normalized (0-1 range)
        if np.max(pixel_boxes) <= 1.0:
            pixel_boxes[:, [0, 2]] *= width   # x coordinates
            pixel_boxes[:, [1, 3]] *= height  # y coordinates

        # Segment from bounding boxes
        masks = self.segment_from_boxes(pixel_boxes, image_rgb)

        # Calculate areas
        include_masks = []
        exclude_masks = []
        include_areas = []
        exclude_areas = []

        for i, mask in enumerate(masks):
            area = np.sum(mask)
            class_id = detections.class_id[i]

            if class_id == 1:  # Include class
                include_masks.append(mask)
                include_areas.append(area)
            else:  # Exclude class
                exclude_masks.append(mask)
                exclude_areas.append(area)

        # Combine masks
        total_include_mask = np.zeros_like(masks[0], dtype=bool) if masks else np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        total_exclude_mask = np.zeros_like(masks[0], dtype=bool) if masks else np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        for mask in include_masks:
            # Convert mask to boolean array before bitwise operation
            mask_bool = mask.astype(bool)
            total_include_mask |= mask_bool

        for mask in exclude_masks:
            # Convert mask to boolean array before bitwise operation
            mask_bool = mask.astype(bool)
            total_exclude_mask |= mask_bool

        return {
            'masks': masks,
            'include_masks': include_masks,
            'exclude_masks': exclude_masks,
            'total_include_mask': total_include_mask,
            'total_exclude_mask': total_exclude_mask,
            'include_areas': include_areas,
            'exclude_areas': exclude_areas,
            'total_include_area': np.sum(total_include_mask),
            'total_exclude_area': np.sum(total_exclude_mask)
        }

    def create_visualization(self, image: np.ndarray,
                           segmentation_result: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of segmentation results.

        Args:
            image: Original image
            segmentation_result: Results from segment_combined

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Overlay include masks (green)
        if 'total_include_mask' in segmentation_result:
            include_mask = segmentation_result['total_include_mask']
            vis_image[include_mask] = vis_image[include_mask] * 0.7 + np.array([0, 255, 0]) * 0.3

        # Overlay exclude masks (red)
        if 'total_exclude_mask' in segmentation_result:
            exclude_mask = segmentation_result['total_exclude_mask']
            vis_image[exclude_mask] = vis_image[exclude_mask] * 0.7 + np.array([0, 0, 255]) * 0.3

        return vis_image

    def create_debug_visualization(self, image: np.ndarray,
                                 segmentation_result: Dict[str, Any],
                                 crop_areas: List[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Create debug visualization with color-coded masks.

        Args:
            image: Original image
            segmentation_result: Results from segment_combined
            crop_areas: List of crop areas as (x1, y1, x2, y2) tuples

        Returns:
            Debug visualization image with color coding:
            - Green: Include areas
            - Red: Exclude areas
            - White: Crop areas
            - Black: Areas that will be cropped out
        """
        height, width = image.shape[:2]
        debug_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Start with black background (areas that will be cropped out)
        debug_image[:] = [0, 0, 0]

        # Add include areas (green)
        if 'total_include_mask' in segmentation_result:
            include_mask = segmentation_result['total_include_mask']
            # Ensure mask has correct dimensions
            if include_mask.ndim == 2 and include_mask.shape == (height, width):
                debug_image[include_mask] = [0, 255, 0]  # Green

        # Add exclude areas (red)
        if 'total_exclude_mask' in segmentation_result:
            exclude_mask = segmentation_result['total_exclude_mask']
            # Ensure mask has correct dimensions
            if exclude_mask.ndim == 2 and exclude_mask.shape == (height, width):
                debug_image[exclude_mask] = [0, 0, 255]  # Red

        # Add crop areas (white)
        if crop_areas:
            for crop_area in crop_areas:
                x1, y1, x2, y2 = crop_area
                debug_image[y1:y2, x1:x2] = [255, 255, 255]  # White

        # If no crop areas specified but we have include areas, show the whole image as crop area
        elif 'total_include_mask' in segmentation_result and np.sum(segmentation_result['total_include_mask']) > 0:
            debug_image[:] = [255, 255, 255]  # White (whole image as crop area)

        return debug_image