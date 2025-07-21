#!/usr/bin/env python3
"""
Tool to analyze debug images and detect color areas.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def analyze_debug_image(image_path: str):
    """Analyze a debug image and report color areas."""
    print(f"\nAnalyzing: {image_path}")
    print("-" * 50)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    height, width = image.shape[:2]
    print(f"Image size: {width} x {height}")

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges (BGR format)
    colors = {
        'black': ([0, 0, 0], [50, 50, 50]),
        'white': ([200, 200, 200], [255, 255, 255]),
        'green': ([0, 100, 0], [100, 255, 100]),
        'red': ([0, 0, 100], [100, 100, 255])
    }

    total_pixels = height * width

    for color_name, (lower, upper) in colors.items():
        # Create mask for this color
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(image, lower, upper)

        # Count pixels
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / total_pixels) * 100

        print(f"{color_name.capitalize()}: {pixel_count:,} pixels ({percentage:.1f}%)")

        # If significant amount, show some details
        if pixel_count > 0:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  Number of {color_name} regions: {len(contours)}")

            # Show largest region
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                x, y, w, h = cv2.boundingRect(largest_contour)
                print(f"  Largest {color_name} region: {w}x{h} at ({x},{y})")

    # Check if image is mostly one color
    dominant_color = None
    max_percentage = 0

    for color_name, (lower, upper) in colors.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(image, lower, upper)
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / total_pixels) * 100

        if percentage > max_percentage:
            max_percentage = percentage
            dominant_color = color_name

    print(f"\nDominant color: {dominant_color} ({max_percentage:.1f}%)")

    if max_percentage > 90:
        print("⚠️  WARNING: Image is mostly one color - this suggests a problem with segmentation!")

def analyze_directory(directory_path: str):
    """Analyze all debug images in a directory."""
    debug_pattern = os.path.join(directory_path, "debug_*.jpg")
    debug_images = glob.glob(debug_pattern)

    if not debug_images:
        print(f"No debug images found in {directory_path}")
        return

    print(f"Found {len(debug_images)} debug images")

    for image_path in sorted(debug_images):
        analyze_debug_image(image_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python analyze_debug_images.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    analyze_directory(directory_path)