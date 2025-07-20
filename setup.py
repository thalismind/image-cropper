#!/usr/bin/env python3
"""
Setup script for Intelligent Image Cropper
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Main setup function."""
    print("Setting up Intelligent Image Cropper...")
    print("="*50)

    # Install required packages
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "supervision>=0.18.0",
        "tqdm>=4.65.0",
        "opencv-contrib-python>=4.8.0"
    ]

    print("Installing Python packages...")
    for package in packages:
        install_package(package)

    # Install Grounding DINO
    print("\nInstalling Grounding DINO...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groundingdino-py"])
        print("✓ Successfully installed groundingdino-py")
    except subprocess.CalledProcessError:
        print("✗ Failed to install groundingdino-py")
        print("Please install manually: pip install groundingdino-py")

    # Install SAM2
    print("\nInstalling SAM2...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "segment-anything"])
        print("✓ Successfully installed segment-anything")
    except subprocess.CalledProcessError:
        print("✗ Failed to install segment-anything")
        print("Please install manually: pip install segment-anything")

    # Create directories
    print("\nCreating directories...")
    directories = ["models", "models/groundingdino", "models/sam2", "demo_output"]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Download models: python download_models.py")
    print("2. Run demo: python demo.py")
    print("3. Process images: python crop_images.py --input_dir ./images --output_dir ./cropped --include 'person' --exclude 'background'")

if __name__ == "__main__":
    main()