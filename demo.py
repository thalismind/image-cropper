#!/usr/bin/env python3
"""
Demo script for Intelligent Image Cropper

This script demonstrates the capabilities of the intelligent cropping system
using sample images and various configurations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from models.intelligent_cropper import IntelligentCropper

def create_sample_image():
    """Create a sample image for demonstration."""
    # Create a simple image with different objects
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200

    # Add a person (rectangle)
    cv2.rectangle(img, (100, 100), (200, 400), (0, 255, 0), -1)
    cv2.putText(img, "PERSON", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add a car (rectangle)
    cv2.rectangle(img, (300, 200), (500, 350), (255, 0, 0), -1)
    cv2.putText(img, "CAR", (350, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add background text (to exclude)
    cv2.putText(img, "BACKGROUND TEXT", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Add a logo (to exclude)
    cv2.circle(img, (700, 100), 50, (0, 0, 255), -1)
    cv2.putText(img, "LOGO", (670, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

def run_demo():
    """Run the demo with sample configurations."""
    print("Intelligent Image Cropper Demo")
    print("="*40)

    # Create sample image
    print("Creating sample image...")
    sample_image = create_sample_image()

    # Save sample image
    os.makedirs("demo_output", exist_ok=True)
    cv2.imwrite("demo_output/sample_image.jpg", sample_image)

    # Initialize cropper
    print("Initializing AI models...")
    cropper = IntelligentCropper(
        confidence_threshold=0.25,  # Lower threshold for demo
        min_area=100,
        padding=30,
        min_crop_size=150
    )

    # Demo configurations
    demo_configs = [
        {
            "name": "Include Person, Exclude Background",
            "include": ["person"],
            "exclude": ["background", "text"]
        },
        {
            "name": "Include Car, Exclude Logo",
            "include": ["car"],
            "exclude": ["logo", "text"]
        },
        {
            "name": "Include Both Person and Car",
            "include": ["person", "car"],
            "exclude": ["text", "logo"]
        }
    ]

    for i, config in enumerate(demo_configs):
        print(f"\nDemo {i+1}: {config['name']}")
        print("-" * 30)

        # Process image
        results = cropper.process_image(
            sample_image,
            config["include"],
            config["exclude"]
        )

        if results['success']:
            # Save cropped images (multiple crops possible)
            for j, cropped_image in enumerate(results['cropped_images']):
                if len(results['cropped_images']) == 1:
                    cropped_path = f"demo_output/cropped_{i+1}.jpg"
                else:
                    cropped_path = f"demo_output/cropped_{i+1}_crop_{j+1}.jpg"
                cv2.imwrite(cropped_path, cropped_image)
                print(f"✓ Cropped image saved: {cropped_path}")

            # Save visualization
            vis_path = f"demo_output/visualization_{i+1}.jpg"
            vis_image = cropper.create_visualization(sample_image, results)
            cv2.imwrite(vis_path, vis_image)
            print(f"✓ Visualization saved: {vis_path}")

            # Print crop area info
            print(f"  Found {len(results['cropped_images'])} crop areas:")
            for j, crop_area in enumerate(results['crop_areas']):
                x1, y1, x2, y2 = crop_area
                print(f"    Crop {j+1}: ({x1}, {y1}) to ({x2}, {y2}) - Size: {x2-x1} x {y2-y1}")
        else:
            print("✗ No valid crop area found")

    print(f"\nDemo complete! Check the 'demo_output' directory for results.")

def run_interactive_demo():
    """Run interactive demo with user input."""
    print("Interactive Intelligent Image Cropper Demo")
    print("="*50)

    # Get image path
    image_path = input("Enter path to image (or press Enter for sample image): ").strip()

    if not image_path:
        print("Using sample image...")
        sample_image = create_sample_image()
        cv2.imwrite("demo_output/sample_image.jpg", sample_image)
        image = sample_image
    else:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            return
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return

    # Get include classes
    include_input = input("Enter classes to include (comma-separated, e.g., 'person,car'): ").strip()
    include_classes = [cls.strip() for cls in include_input.split(",") if cls.strip()]

    if not include_classes:
        print("Error: At least one include class must be specified")
        return

    # Get exclude classes
    exclude_input = input("Enter classes to exclude (comma-separated, e.g., 'background,text'): ").strip()
    exclude_classes = [cls.strip() for cls in exclude_input.split(",") if cls.strip()]

    # Initialize cropper
    print("Initializing AI models...")
    cropper = IntelligentCropper(
        confidence_threshold=0.3,
        min_area=100,
        padding=30,
        min_crop_size=150
    )

    # Process image
    print("Processing image...")
    results = cropper.process_image(image, include_classes, exclude_classes)

    if results['success']:
        # Save results
        os.makedirs("demo_output", exist_ok=True)

        # Save cropped images (multiple crops possible)
        for j, cropped_image in enumerate(results['cropped_images']):
            if len(results['cropped_images']) == 1:
                cropped_path = "demo_output/interactive_cropped.jpg"
            else:
                cropped_path = f"demo_output/interactive_cropped_crop_{j+1}.jpg"
            cv2.imwrite(cropped_path, cropped_image)
            print(f"✓ Cropped image saved: {cropped_path}")

        vis_path = "demo_output/interactive_visualization.jpg"
        vis_image = cropper.create_visualization(image, results)
        cv2.imwrite(vis_path, vis_image)
        print(f"✓ Visualization saved: {vis_path}")

        # Print info
        print(f"  Found {len(results['cropped_images'])} crop areas:")
        for j, crop_area in enumerate(results['crop_areas']):
            x1, y1, x2, y2 = crop_area
            print(f"    Crop {j+1}: ({x1}, {y1}) to ({x2}, {y2}) - Size: {x2-x1} x {y2-y1}")

    else:
        print("✗ No valid crop area found")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Intelligent Image Cropper Demo")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive demo")

    args = parser.parse_args()

    if args.interactive:
        run_interactive_demo()
    else:
        run_demo()

if __name__ == "__main__":
    main()