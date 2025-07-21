#!/usr/bin/env python3
"""
Example usage of Intelligent Image Cropper

This script demonstrates various use cases and configurations
for the intelligent image cropping system.
"""

import os
import cv2
import numpy as np
from pathlib import Path

from models.intelligent_cropper import IntelligentCropper

def create_example_images():
    """Create example images for demonstration."""
    os.makedirs("example_images", exist_ok=True)

    # Example 1: Portrait photo with background
    img1 = np.ones((600, 800, 3), dtype=np.uint8) * 180
    # Add person
    cv2.rectangle(img1, (300, 100), (500, 500), (0, 255, 0), -1)
    cv2.putText(img1, "PERSON", (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # Add background elements
    cv2.rectangle(img1, (50, 50), (200, 150), (100, 100, 100), -1)
    cv2.putText(img1, "BACKGROUND", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite("example_images/portrait.jpg", img1)

    # Example 2: Street scene with multiple objects
    img2 = np.ones((600, 800, 3), dtype=np.uint8) * 200
    # Add cars
    cv2.rectangle(img2, (100, 200), (300, 350), (255, 0, 0), -1)
    cv2.putText(img2, "CAR", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(img2, (400, 250), (600, 400), (0, 0, 255), -1)
    cv2.putText(img2, "CAR", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Add people
    cv2.rectangle(img2, (50, 100), (150, 300), (0, 255, 0), -1)
    cv2.putText(img2, "PERSON", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Add text to exclude
    cv2.putText(img2, "STREET SIGN", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite("example_images/street_scene.jpg", img2)

    # Example 3: Product photo with logo
    img3 = np.ones((500, 700, 3), dtype=np.uint8) * 220
    # Add product
    cv2.rectangle(img3, (200, 100), (500, 400), (255, 255, 0), -1)
    cv2.putText(img3, "PRODUCT", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # Add logo to exclude
    cv2.circle(img3, (100, 100), 40, (255, 0, 0), -1)
    cv2.putText(img3, "LOGO", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite("example_images/product.jpg", img3)

    print("✓ Created example images in 'example_images' directory")

def run_examples():
    """Run various example configurations."""
    print("Intelligent Image Cropper - Example Usage")
    print("="*50)

    # Create example images
    create_example_images()

    # Initialize cropper
    print("\nInitializing AI models...")
    cropper = IntelligentCropper(
        confidence_threshold=0.25,
        min_area=100,
        padding=30
    )

    # Example configurations
    examples = [
        {
            "name": "Portrait Cropping",
            "image": "example_images/portrait.jpg",
            "include": ["person"],
            "exclude": ["background"],
            "description": "Crop to include person while excluding background"
        },
        {
            "name": "Street Scene - Cars Only",
            "image": "example_images/street_scene.jpg",
            "include": ["car"],
            "exclude": ["text", "person"],
            "description": "Focus on cars, exclude people and text"
        },
        {
            "name": "Street Scene - People Only",
            "image": "example_images/street_scene.jpg",
            "include": ["person"],
            "exclude": ["car", "text"],
            "description": "Focus on people, exclude cars and text"
        },
        {
            "name": "Product Photography",
            "image": "example_images/product.jpg",
            "include": ["product"],
            "exclude": ["logo"],
            "description": "Crop product while excluding logo"
        }
    ]

    os.makedirs("example_output", exist_ok=True)

    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example['name']}")
        print(f"Description: {example['description']}")
        print("-" * 40)

        # Load image
        image = cv2.imread(example['image'])
        if image is None:
            print(f"✗ Could not load image: {example['image']}")
            continue

        # Process image
        results = cropper.process_image(
            image,
            example['include'],
            example['exclude']
        )

        if results['success']:
            # Save cropped image
            cropped_path = f"example_output/cropped_{i+1}.jpg"
            cv2.imwrite(cropped_path, results['cropped_image'])
            print(f"✓ Cropped image saved: {cropped_path}")

            # Save visualization
            vis_path = f"example_output/visualization_{i+1}.jpg"
            vis_image = cropper.create_visualization(image, results)
            cv2.imwrite(vis_path, vis_image)
            print(f"✓ Visualization saved: {vis_path}")

            # Print crop info
            x1, y1, x2, y2 = results['crop_area']
            print(f"  Crop area: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Crop size: {x2-x1} x {y2-y1}")

            # Print detection info
            seg_results = results['segmentation_results']
            print(f"  Include area: {seg_results.get('total_include_area', 0)} pixels")
            print(f"  Exclude area: {seg_results.get('total_exclude_area', 0)} pixels")
        else:
            print("✗ No valid crop area found")

    print(f"\nExample processing complete!")
    print("Check the 'example_output' directory for results.")

def show_usage_examples():
    """Show command line usage examples."""
    print("\nCommand Line Usage Examples:")
    print("="*40)

    examples = [
        {
            "description": "Basic portrait cropping",
            "command": "python crop_images.py --input_dir ./photos --output_dir ./cropped --include 'person' --exclude 'background'"
        },
        {
            "description": "Product photography with logo exclusion",
            "command": "python crop_images.py --input_dir ./products --output_dir ./clean --include 'product' --exclude 'logo,text' --confidence 0.6"
        },
        {
            "description": "Street scene focusing on people",
            "command": "python crop_images.py --input_dir ./street --output_dir ./people --include 'person' --exclude 'car,text' --padding 100"
        },
        {
            "description": "Multiple object detection",
            "command": "python crop_images.py --input_dir ./images --output_dir ./cropped --include 'person,car,animal' --exclude 'text,logo' --save_visualizations"
        },
        {
            "description": "Test with limited images",
            "command": "python crop_images.py --input_dir ./test --output_dir ./results --include 'person' --max_images 5"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print(f"   {example['command']}")

def main():
    """Main function."""
    print("Intelligent Image Cropper - Example Usage")
    print("="*50)

    # Run examples
    run_examples()

    # Show usage examples
    show_usage_examples()

    print("\n" + "="*50)
    print("For more information, see README.md")
    print("For testing, run: python test_installation.py")
    print("For demo, run: python demo.py")

if __name__ == "__main__":
    main()