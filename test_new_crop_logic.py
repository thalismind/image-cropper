#!/usr/bin/env python3
"""
Test script for the new crop algorithm logic.
"""

import numpy as np
import cv2
import os
from models.intelligent_cropper import IntelligentCropper

def create_test_images():
    """Create test images for different scenarios."""
    os.makedirs("test_images", exist_ok=True)

    # Test 1: No exclusions - should keep whole image
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.rectangle(img1, (100, 100), (200, 300), (0, 255, 0), -1)  # Person
    cv2.putText(img1, "PERSON", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite("test_images/no_exclusions.jpg", img1)

    # Test 2: Multiple inclusions - should create separate crops
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.rectangle(img2, (50, 100), (150, 300), (0, 255, 0), -1)  # Person 1
    cv2.putText(img2, "PERSON1", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.rectangle(img2, (350, 100), (450, 300), (0, 255, 0), -1)  # Person 2
    cv2.putText(img2, "PERSON2", (360, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.rectangle(img2, (200, 50), (300, 100), (255, 0, 0), -1)  # Text to exclude
    cv2.putText(img2, "TEXT", (210, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite("test_images/multiple_inclusions.jpg", img2)

    # Test 3: Small inclusions - should expand to minimum size
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.rectangle(img3, (250, 150), (270, 170), (0, 255, 0), -1)  # Small person
    cv2.putText(img3, "SMALL", (240, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite("test_images/small_inclusion.jpg", img3)

    print("✓ Created test images")

def test_crop_logic():
    """Test the new crop algorithm logic."""
    print("Testing New Crop Algorithm Logic")
    print("="*40)

    # Initialize cropper with small minimum crop size for testing
    cropper = IntelligentCropper(
        confidence_threshold=0.3,
        min_area=50,
        padding=20,
        min_crop_size=100
    )

    # Test scenarios
    test_scenarios = [
        {
            "name": "No Exclusions - Should Keep Whole Image",
            "image": "test_images/no_exclusions.jpg",
            "include": ["person"],
            "exclude": [],
            "expected_crops": 1,
            "expected_whole_image": True
        },
        {
            "name": "Multiple Inclusions - Should Create Separate Crops",
            "image": "test_images/multiple_inclusions.jpg",
            "include": ["person"],
            "exclude": ["text"],
            "expected_crops": 2,
            "expected_whole_image": False
        },
        {
            "name": "Small Inclusion - Should Expand to Minimum Size",
            "image": "test_images/small_inclusion.jpg",
            "include": ["person"],
            "exclude": [],
            "expected_crops": 1,
            "expected_whole_image": False
        }
    ]

    os.makedirs("test_output", exist_ok=True)

    for i, scenario in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {scenario['name']}")
        print("-" * 40)

        # Load image
        image = cv2.imread(scenario['image'])
        if image is None:
            print(f"✗ Could not load image: {scenario['image']}")
            continue

        # Process image
        results = cropper.process_image(
            image,
            scenario['include'],
            scenario['exclude']
        )

        # Check results
        num_crops = len(results['cropped_images'])
        print(f"  Found {num_crops} crop(s)")

        if results['success']:
            # Save results
            for j, cropped_image in enumerate(results['cropped_images']):
                output_path = f"test_output/test_{i+1}_crop_{j+1}.jpg"
                cv2.imwrite(output_path, cropped_image)
                print(f"  ✓ Saved crop {j+1}: {output_path}")

            # Check if whole image was kept
            if scenario['expected_whole_image']:
                height, width = image.shape[:2]
                whole_image_crop = (0, 0, width, height)
                if whole_image_crop in results['crop_areas']:
                    print("  ✓ Correctly kept whole image")
                else:
                    print("  ✗ Expected whole image but got different crop")
            else:
                print("  ✓ Created individual crop(s)")

            # Check number of crops
            if num_crops == scenario['expected_crops']:
                print(f"  ✓ Correct number of crops: {num_crops}")
            else:
                print(f"  ✗ Expected {scenario['expected_crops']} crops, got {num_crops}")

        else:
            print("  ✗ No valid crop areas found")

    print(f"\nTest complete! Check the 'test_output' directory for results.")

def main():
    """Main function."""
    print("Testing New Crop Algorithm")
    print("="*30)

    # Create test images
    create_test_images()

    # Run tests
    test_crop_logic()

if __name__ == "__main__":
    main()