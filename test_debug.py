#!/usr/bin/env python3
"""
Test script for debug visualization functionality.
"""

import numpy as np
import cv2
import os
from models.intelligent_cropper import IntelligentCropper

def create_test_image():
    """Create a simple test image with some objects."""
    # Create a 400x600 test image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Gray background

    # Add a person (green rectangle)
    cv2.rectangle(image, (100, 100), (200, 300), (0, 255, 0), -1)
    cv2.putText(image, "PERSON", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add text to exclude (red rectangle)
    cv2.rectangle(image, (300, 50), (400, 100), (0, 0, 255), -1)
    cv2.putText(image, "TEXT", (310, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def test_debug_visualization():
    """Test the debug visualization functionality."""
    print("Testing Debug Visualization")
    print("="*40)

    # Create test image
    test_image = create_test_image()

    # Save test image
    os.makedirs("debug_test", exist_ok=True)
    cv2.imwrite("debug_test/test_image.jpg", test_image)
    print("✓ Created test image")

    # Initialize cropper
    cropper = IntelligentCropper(
        confidence_threshold=0.3,
        min_area=100,
        padding=20,
        min_crop_size=150
    )

    # Process image
    results = cropper.process_image(test_image, ["person"], ["text"])

    print(f"Processing results:")
    print(f"  Success: {results['success']}")
    print(f"  Number of crops: {len(results['cropped_images'])}")

    # Create debug visualization
    debug_image = cropper.create_debug_visualization(test_image, results)
    cv2.imwrite("debug_test/debug_visualization.jpg", debug_image)
    print("✓ Created debug visualization")

    # Print color coding legend
    print("\nDebug visualization color coding:")
    print("  Green: Include areas (person)")
    print("  Red: Exclude areas (text)")
    print("  White: Crop areas")
    print("  Black: Areas that will be cropped out")

    print(f"\nTest complete! Check 'debug_test/' directory for results.")

if __name__ == "__main__":
    test_debug_visualization()