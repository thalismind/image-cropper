#!/usr/bin/env python3
"""
Debug script to analyze detection and segmentation results.
"""

import cv2
import numpy as np
import os
from models.intelligent_cropper import IntelligentCropper

def debug_single_image(image_path: str):
    """Debug a single image's detection and segmentation."""
    print(f"\nDebugging: {image_path}")
    print("="*60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    # Initialize cropper
    cropper = IntelligentCropper(
        confidence_threshold=0.3,  # Lower threshold for debugging
        min_area=100,
        padding=20,
        min_crop_size=150
    )

    # Process image
    results = cropper.process_image(image, ["person", "woman"], ["text", "watermark"])

    print(f"\nProcessing Results:")
    print(f"  Success: {results['success']}")
    print(f"  Number of crops: {len(results['cropped_images'])}")

    # Analyze detection results
    detection_results = results['detection_results']
    print(f"\nDetection Results:")
    print(f"  Include detections: {len(detection_results.get('include_detections', []))}")
    print(f"  Exclude detections: {len(detection_results.get('exclude_detections', []))}")
    print(f"  All detections: {len(detection_results.get('all_detections', []))}")

    # Analyze segmentation results
    seg_results = results['segmentation_results']
    print(f"\nSegmentation Results:")
    print(f"  Total masks: {len(seg_results.get('masks', []))}")
    print(f"  Include masks: {len(seg_results.get('include_masks', []))}")
    print(f"  Exclude masks: {len(seg_results.get('exclude_masks', []))}")
    print(f"  Total include area: {seg_results.get('total_include_area', 0)}")
    print(f"  Total exclude area: {seg_results.get('total_exclude_area', 0)}")

    # Check if masks exist and their properties
    if 'total_include_mask' in seg_results:
        include_mask = seg_results['total_include_mask']
        print(f"  Include mask shape: {include_mask.shape}")
        print(f"  Include mask sum: {np.sum(include_mask)}")
        print(f"  Include mask dtype: {include_mask.dtype}")

    if 'total_exclude_mask' in seg_results:
        exclude_mask = seg_results['total_exclude_mask']
        print(f"  Exclude mask shape: {exclude_mask.shape}")
        print(f"  Exclude mask sum: {np.sum(exclude_mask)}")
        print(f"  Exclude mask dtype: {exclude_mask.dtype}")

    # Create debug visualization
    debug_image = cropper.create_debug_visualization(image, results)

    # Save debug image
    output_dir = "debug_analysis"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_path = os.path.join(output_dir, f"debug_{base_name}.jpg")
    cv2.imwrite(debug_path, debug_image)
    print(f"\nDebug image saved: {debug_path}")

    # Also save original image for comparison
    orig_path = os.path.join(output_dir, f"original_{base_name}.jpg")
    cv2.imwrite(orig_path, image)
    print(f"Original image saved: {orig_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python debug_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_single_image(image_path)