#!/usr/bin/env python3
"""
Download required model files for the intelligent image cropper.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

def download_file(url, filename):
    """Download a file from URL to filename."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ["models", "models/groundingdino", "models/sam2"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_grounding_dino():
    """Download Grounding DINO model files."""
    print("Downloading Grounding DINO models...")

    # Grounding DINO config and weights
    grounding_dino_urls = {
        "models/groundingdino/groundingdino_swint_ogc.py":
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "models/groundingdino/groundingdino_swint_ogc.pth":
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    }

    for filename, url in grounding_dino_urls.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists, skipping...")

def download_sam2():
    """Download SAM2.1 model files."""
    print("Downloading SAM2.1 models...")

    # Also download original SAM model for compatibility
    sam2_urls={
        "models/sam2/sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }

    for filename, url in sam2_urls.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists, skipping...")

def install_grounding_dino():
    """Install Grounding DINO package."""
    print("Installing Grounding DINO...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groundingdino-py"])
        print("Grounding DINO installed successfully")
    except Exception as e:
        print(f"Error installing Grounding DINO: {e}")
        print("Please install manually: pip install groundingdino-py")

def install_sam2():
    """Install SAM2 package."""
    print("Installing SAM2...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "segment-anything"])
        print("SAM2 installed successfully")
    except Exception as e:
        print(f"Error installing SAM2: {e}")
        print("Please install manually: pip install segment-anything")

def main():
    """Main download function."""
    print("Setting up models for Intelligent Image Cropper...")

    # Create directories
    create_directories()

    # Install packages
    install_grounding_dino()
    install_sam2()

    # Download model files
    download_grounding_dino()
    download_sam2()

    print("\nSetup complete! You can now run the image cropper.")
    print("Example usage:")
    print("python crop_images.py --input_dir ./images --output_dir ./cropped --include 'person' --exclude 'background'")

if __name__ == "__main__":
    main()