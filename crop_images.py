#!/usr/bin/env python3
"""
Intelligent Image Cropper - Main Script

Uses Segment Anything v2 and Grounding DINO to intelligently crop images
based on AI-detected areas, maximizing inclusion areas while excluding unwanted regions.
"""

import argparse
import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.intelligent_cropper import IntelligentCropper

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Intelligent Image Cropper using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crop_images.py --input_dir ./images --output_dir ./cropped --include "person" --exclude "background"
  python crop_images.py --input_dir ./photos --include "car,person" --exclude "text,logo" --confidence 0.6
  python crop_images.py --input_dir ./dataset --include "animal" --exclude "human" --padding 100
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cropped images"
    )

    parser.add_argument(
        "--include",
        type=str,
        required=True,
        help="Comma-separated keywords for areas to include"
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated keywords for areas to exclude"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=1000,
        help="Minimum area for detected objects (default: 1000)"
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=50,
        help="Padding around crop area in pixels (default: 50)"
    )

    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=200,
        help="Minimum size for individual crop areas (default: 200)"
    )

    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tiff",
        help="Glob pattern for image files (default: *.jpg,*.jpeg,*.png,*.bmp,*.tiff)"
    )

    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images showing detection and segmentation results"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images with color-coded masks (green=include, red=exclude, white=crop area, black=cropped out)"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )

    return parser.parse_args()

def get_image_files(input_dir: str, glob_pattern: str) -> List[str]:
    """Get list of image files matching the glob pattern."""
    patterns = glob_pattern.split(",")
    image_files = []

    for pattern in patterns:
        pattern = pattern.strip()
        files = glob.glob(os.path.join(input_dir, pattern))
        image_files.extend(files)

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    return image_files

def process_single_image(image_path: str, cropper: IntelligentCropper,
                        include_classes: List[str], exclude_classes: List[str],
                        output_dir: str, save_visualizations: bool = False, debug: bool = False) -> bool:
    """
    Process a single image.

    Args:
        image_path: Path to input image
        cropper: IntelligentCropper instance
        include_classes: List of classes to include
        exclude_classes: List of classes to exclude
        output_dir: Output directory
        save_visualizations: Whether to save visualization images
        debug: Whether to save debug images with color-coded masks

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        # Process image
        results = cropper.process_image(image, include_classes, exclude_classes)

        if not results['success']:
            print(f"Warning: No valid crop area found for {image_path}")
            # Even if no crop area found, save debug image if requested
            if debug:
                debug_image = cropper.create_debug_visualization(image, results)
                debug_path = os.path.join(output_dir, f"debug_{os.path.basename(image_path)}")
                cv2.imwrite(debug_path, debug_image)
            return False

        # Save cropped images (multiple crops possible)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        extension = os.path.splitext(image_path)[1]

        for i, cropped_image in enumerate(results['cropped_images']):
            if len(results['cropped_images']) == 1:
                # Single crop, use original filename
                output_path = os.path.join(output_dir, os.path.basename(image_path))
            else:
                # Multiple crops, add index to filename
                output_path = os.path.join(output_dir, f"{base_name}_crop_{i+1}{extension}")

            cv2.imwrite(output_path, cropped_image)

        # Save visualization if requested
        if save_visualizations:
            vis_image = cropper.create_visualization(image, results)
            vis_path = os.path.join(output_dir, f"vis_{os.path.basename(image_path)}")
            cv2.imwrite(vis_path, vis_image)

        # Save debug image if requested
        if debug:
            debug_image = cropper.create_debug_visualization(image, results)
            debug_path = os.path.join(output_dir, f"debug_{os.path.basename(image_path)}")
            cv2.imwrite(debug_path, debug_image)

        return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    """Main function."""
    args = parse_arguments()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse include and exclude classes
    include_classes = [cls.strip() for cls in args.include.split(",") if cls.strip()]
    exclude_classes = [cls.strip() for cls in args.exclude.split(",") if cls.strip()]

    if not include_classes:
        print("Error: At least one include class must be specified")
        return

    print(f"Include classes: {include_classes}")
    print(f"Exclude classes: {exclude_classes}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Minimum area: {args.min_area}")
    print(f"Padding: {args.padding}")
    print(f"Minimum crop size: {args.min_crop_size}")

    # Initialize cropper
    print("Initializing AI models...")
    cropper = IntelligentCropper(
        confidence_threshold=args.confidence,
        min_area=args.min_area,
        padding=args.padding,
        min_crop_size=args.min_crop_size
    )

    # Get image files
    image_files = get_image_files(args.input_dir, args.glob_pattern)

    if not image_files:
        print(f"No image files found in {args.input_dir} matching pattern {args.glob_pattern}")
        return

    print(f"Found {len(image_files)} images to process")

    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Processing first {len(image_files)} images (max_images={args.max_images})")

    # Process images
    successful = 0
    failed = 0

    print("Processing images...")
    for image_path in tqdm(image_files, desc="Processing images"):
        success = process_single_image(
            image_path, cropper, include_classes, exclude_classes,
            args.output_dir, args.save_visualizations, args.debug
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(image_files)*100:.1f}%")
    print(f"Output directory: {args.output_dir}")

    if args.save_visualizations:
        print("Visualization images saved with 'vis_' prefix")

    if args.debug:
        print("Debug images saved with 'debug_' prefix")

if __name__ == "__main__":
    main()