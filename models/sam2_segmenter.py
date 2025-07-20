"""
SAM2 segmenter for precise object segmentation.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import supervision as sv
from segment_anything import SamPredictor, sam_model_registry
import os

class SAM2Segmenter:
    """SAM2.1 segmenter for precise object segmentation."""

    def __init__(self, model_path: str = "models/sam2/sam_vit_h_4b8939.pth",
                 model_type: str = "vit_h",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SAM2.1 segmenter.

        Args:
            model_path: Path to SAM2.1 model checkpoint
            model_type: Type of SAM2.1 model (hiera_tiny, hiera_small, hiera_base_plus, hiera_large)
            device: Device to run inference on
        """
        self.device = device
        self.model_type = model_type
        self.predictor = self._load_model(model_path)

    def _load_model(self, model_path: str) -> SamPredictor:
        """Load SAM2.1 model."""
        try:
            # Load SAM2.1 model
            sam = sam_model_registry[self.model_type](checkpoint=model_path)
            sam.to(device=self.device)

            # Create predictor
            predictor = SamPredictor(sam)

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

            # Get mask
            mask, _, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )

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

        # Get mask
        mask, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

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

        # Segment from bounding boxes
        masks = self.segment_from_boxes(detections.xyxy, image_rgb)

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
            total_include_mask |= mask

        for mask in exclude_masks:
            total_exclude_mask |= mask

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