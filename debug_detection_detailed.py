#!/usr/bin/env python3
"""
Detailed debug script to analyze detection and segmentation results.
"""

import cv2
import numpy as np
import os
from models.intelligent_cropper import IntelligentCropper
from models.grounding_dino_detector import GroundingDINODetector

def debug_detection_detailed(image_path: str):
    """Debug detection and segmentation in detail."""
    print(f"\nDetailed Debugging: {image_path}")
    print("="*60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    # Test detection directly
    detector = GroundingDINODetector()

    # Test include classes
    print(f"\nTesting include classes: ['person', 'woman']")
    include_results = detector.detect_multiple_classes(image, ["person", "woman"], [], 0.3)
    print(f"  Include detections: {len(include_results.get('include', {}))}")
    for class_name, data in include_results.get('include', {}).items():
        detections = data['detections']
        print(f"    {class_name}: {len(detections)} detections")
        for i, (xyxy, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            print(f"      {class_name} {i}: conf={conf:.3f}, box={xyxy}")

    # Test exclude classes
    print(f"\nTesting exclude classes: ['text', 'watermark']")
    exclude_results = detector.detect_multiple_classes(image, [], ["text", "watermark"], 0.3)
    print(f"  Exclude detections: {len(exclude_results.get('exclude', {}))}")
    for class_name, data in exclude_results.get('exclude', {}).items():
        detections = data['detections']
        print(f"    {class_name}: {len(detections)} detections")
        for i, (xyxy, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            print(f"      {class_name} {i}: conf={conf:.3f}, box={xyxy}")

    # Test combined detection
    print(f"\nTesting combined detection")
    combined_results = detector.detect_multiple_classes(image, ["person", "woman"], ["text", "watermark"], 0.3)
    print(f"  All detections: {len(combined_results.get('all_detections', []))}")
    all_detections = combined_results.get('all_detections', [])
    if len(all_detections) > 0:
        for i, (xyxy, conf, class_id) in enumerate(zip(all_detections.xyxy, all_detections.confidence, all_detections.class_id)):
            class_type = "include" if class_id == 1 else "exclude"
            print(f"    Detection {i}: {class_type}, conf={conf:.3f}, box={xyxy}")

    # Test with lower confidence
    print(f"\nTesting with lower confidence (0.1)")
    low_conf_results = detector.detect_multiple_classes(image, ["person", "woman"], ["text", "watermark"], 0.1)
    print(f"  All detections (low conf): {len(low_conf_results.get('all_detections', []))}")
    all_detections = low_conf_results.get('all_detections', [])
    if len(all_detections) > 0:
        for i, (xyxy, conf, class_id) in enumerate(zip(all_detections.xyxy, all_detections.confidence, all_detections.class_id)):
            class_type = "include" if class_id == 1 else "exclude"
            print(f"    Detection {i}: {class_type}, conf={conf:.3f}, box={xyxy}")

    # Test individual classes
    print(f"\nTesting individual classes:")
    for class_name in ["person", "woman", "text", "watermark"]:
        single_results = detector.detect_multiple_classes(image, [class_name], [], 0.1)
        detections = single_results.get('include', {}).get(class_name, {}).get('detections', [])
        print(f"  {class_name}: {len(detections)} detections")
        for i, (xyxy, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            print(f"    {class_name} {i}: conf={conf:.3f}, box={xyxy}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python debug_detection_detailed.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_detection_detailed(image_path)