"""
Intelligent cropper that combines detection and segmentation for optimal image cropping.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import supervision as sv
from .grounding_dino_detector import GroundingDINODetector
from .sam2_segmenter import SAM2Segmenter

class IntelligentCropper:
    """Intelligent cropper that maximizes included areas while excluding unwanted regions."""

    def __init__(self, confidence_threshold: float = 0.5,
                 min_area: int = 1000, padding: int = 50, min_crop_size: int = 200):
        """
        Initialize intelligent cropper.

        Args:
            confidence_threshold: Minimum confidence for detections
            min_area: Minimum area for detected objects
            padding: Padding around crop area in pixels
            min_crop_size: Minimum size for individual crop areas (width/height)
        """
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.padding = padding
        self.min_crop_size = min_crop_size

        # Initialize models
        self.detector = GroundingDINODetector()
        self.segmenter = SAM2Segmenter()

    def process_image(self, image: np.ndarray,
                     include_classes: List[str],
                     exclude_classes: List[str]) -> Dict[str, Any]:
        """
        Process image to find optimal crop areas.

        Args:
            image: Input image as numpy array
            include_classes: List of classes to include
            exclude_classes: List of classes to exclude

        Returns:
            Dictionary with processing results
        """
        # Detect objects
        detection_results = self.detector.detect_multiple_classes(
            image, include_classes, exclude_classes, self.confidence_threshold
        )

        # Segment detected objects
        segmentation_results = self.segmenter.segment_combined(
            detection_results['all_detections'], image
        )

        # Find optimal crop areas
        crop_areas = self._find_optimal_crop_areas(
            image, segmentation_results, include_classes, exclude_classes
        )

        # Create cropped images
        cropped_images = []
        if crop_areas:
            for crop_area in crop_areas:
                cropped_image = self._crop_image(image, crop_area)
                cropped_images.append(cropped_image)

        return {
            'detection_results': detection_results,
            'segmentation_results': segmentation_results,
            'crop_areas': crop_areas,
            'cropped_images': cropped_images,
            'success': len(crop_areas) > 0
        }

    def _find_optimal_crop_areas(self, image: np.ndarray,
                                segmentation_results: Dict[str, Any],
                                include_classes: List[str],
                                exclude_classes: List[str]) -> List[Tuple[int, int, int, int]]:
        """
        Find optimal crop areas that maximize included areas while excluding unwanted regions.

        Args:
            image: Input image
            segmentation_results: Results from segmentation
            include_classes: Classes to include
            exclude_classes: Classes to exclude

        Returns:
            List of optimal crop areas as (x1, y1, x2, y2) tuples
        """
        height, width = image.shape[:2]

        # Get masks
        include_mask = segmentation_results.get('total_include_mask', np.zeros((height, width), dtype=bool))
        exclude_mask = segmentation_results.get('total_exclude_mask', np.zeros((height, width), dtype=bool))

        # If no include areas found, return empty list
        if np.sum(include_mask) == 0:
            return []

        # If no exclusion areas found, keep the whole image
        if np.sum(exclude_mask) == 0:
            return [(0, 0, width, height)]

        # Find individual inclusion areas
        crop_areas = self._find_individual_crop_areas(
            image, include_mask, exclude_mask
        )

        return crop_areas

    def _find_individual_crop_areas(self, image: np.ndarray,
                                   include_mask: np.ndarray,
                                   exclude_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find individual crop areas for each inclusion region.

        Args:
            image: Input image
            include_mask: Mask of areas to include
            exclude_mask: Mask of areas to exclude

        Returns:
            List of crop areas as (x1, y1, x2, y2) tuples
        """
        height, width = image.shape[:2]
        crop_areas = []

        # Find connected components in include mask
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(include_mask)

        if num_features == 0:
            return []

        # Process each connected component
        for label in range(1, num_features + 1):
            # Create mask for this component
            component_mask = (labeled_mask == label)

            # Find bounding box of this component
            coords = np.where(component_mask)
            if len(coords[0]) == 0:
                continue

            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])

            # Add padding
            min_x = max(0, min_x - self.padding)
            min_y = max(0, min_y - self.padding)
            max_x = min(width, max_x + self.padding)
            max_y = min(height, max_y + self.padding)

            # Check minimum crop size
            crop_width = max_x - min_x
            crop_height = max_y - min_y

            if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                # Expand crop to meet minimum size
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2

                half_size = self.min_crop_size // 2
                min_x = max(0, center_x - half_size)
                min_y = max(0, center_y - half_size)
                max_x = min(width, center_x + half_size)
                max_y = min(height, center_y + half_size)

            # Check if this crop area overlaps with exclusion areas
            crop_exclude_mask = exclude_mask[min_y:max_y, min_x:max_x]

            if np.sum(crop_exclude_mask) > 0:
                # Try to adjust crop to minimize exclusions
                adjusted_crop = self._adjust_crop_for_exclusions(
                    image, component_mask, exclude_mask, (min_x, min_y, max_x, max_y)
                )
                if adjusted_crop is not None:
                    crop_areas.append(adjusted_crop)
                else:
                    # If adjustment fails, skip this crop area
                    continue
            else:
                # No exclusions in this area, use as is
                crop_areas.append((min_x, min_y, max_x, max_y))

        return crop_areas

    def _adjust_crop_for_exclusions(self, image: np.ndarray,
                                   include_mask: np.ndarray,
                                   exclude_mask: np.ndarray,
                                   initial_crop: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Adjust crop area to minimize excluded regions.

        Args:
            image: Input image
            include_mask: Mask of areas to include
            exclude_mask: Mask of areas to exclude
            initial_crop: Initial crop area (x1, y1, x2, y2)

        Returns:
            Adjusted crop area or None if no good adjustment found
        """
        min_x, min_y, max_x, max_y = initial_crop
        height, width = image.shape[:2]

        # Get the crop region
        crop_include = include_mask[min_y:max_y, min_x:max_x]
        crop_exclude = exclude_mask[min_y:max_y, min_x:max_x]

        # Calculate the center of include areas
        include_coords = np.where(crop_include)
        if len(include_coords[0]) == 0:
            return None

        center_y = np.mean(include_coords[0])
        center_x = np.mean(include_coords[1])

        # Try different crop sizes centered on the include areas
        crop_width = max_x - min_x
        crop_height = max_y - min_y

        # Try reducing the crop size
        for scale in [0.9, 0.8, 0.7, 0.6]:
            new_width = int(crop_width * scale)
            new_height = int(crop_height * scale)

            # Calculate new crop bounds
            new_min_x = max(0, min_x + (crop_width - new_width) // 2)
            new_min_y = max(0, min_y + (crop_height - new_height) // 2)
            new_max_x = min(width, new_min_x + new_width)
            new_max_y = min(height, new_min_y + new_height)

            # Check if this crop area has fewer exclusions
            new_crop_exclude = exclude_mask[new_min_y:new_max_y, new_min_x:new_max_x]
            new_crop_include = include_mask[new_min_y:new_max_y, new_min_x:new_max_x]

            # If we have include areas and fewer exclude areas, use this crop
            if np.sum(new_crop_include) > 0 and np.sum(new_crop_exclude) < np.sum(crop_exclude):
                return (new_min_x, new_min_y, new_max_x, new_max_y)

        return None

    def _crop_image(self, image: np.ndarray, crop_area: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image to the specified area.

        Args:
            image: Input image
            crop_area: Crop area as (x1, y1, x2, y2)

        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = crop_area
        return image[y1:y2, x1:x2]

    def create_visualization(self, image: np.ndarray,
                           processing_results: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of processing results.

        Args:
            image: Original image
            processing_results: Results from process_image

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Add segmentation visualization
        if 'segmentation_results' in processing_results:
            vis_image = self.segmenter.create_visualization(vis_image, processing_results['segmentation_results'])

        # Add crop area visualization
        crop_areas = processing_results.get('crop_areas', [])
        for i, crop_area in enumerate(crop_areas):
            x1, y1, x2, y2 = crop_area
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(vis_image, f'Crop {i+1}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return vis_image

    def create_debug_visualization(self, image: np.ndarray,
                                 processing_results: Dict[str, Any]) -> np.ndarray:
        """
        Create debug visualization with color-coded masks.

        Args:
            image: Original image
            processing_results: Results from process_image

        Returns:
            Debug visualization image with color coding:
            - Green: Include areas
            - Red: Exclude areas
            - White: Crop areas
            - Black: Areas that will be cropped out
        """
        crop_areas = processing_results.get('crop_areas', [])
        segmentation_results = processing_results.get('segmentation_results', {})

        return self.segmenter.create_debug_visualization(image, segmentation_results, crop_areas)