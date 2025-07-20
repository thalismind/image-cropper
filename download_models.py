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

    # Grounding DINO config and weights (SwinB version)
    grounding_dino_urls = {
        "models/groundingdino/groundingdino_swinb_cogcoor.py":
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "models/groundingdino/groundingdino_swinb_cogcoor.pth":
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    }

    for filename, url in grounding_dino_urls.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists, skipping...")

def download_sam2():
    """Download SAM2.1 model files."""
    print("Downloading SAM2.1 models...")

    # Define the base URL for SAM 2.1 checkpoints
    SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

    # SAM2.1 model URLs
    sam2_urls = {
        "models/sam2/sam2.1_hiera_tiny.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt",
        "models/sam2/sam2.1_hiera_small.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt",
        "models/sam2/sam2.1_hiera_base_plus.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt",
        "models/sam2/sam2.1_hiera_large.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
    }

    # Also download original SAM model for compatibility
    sam2_urls["models/sam2/sam_vit_h_4b8939.pth"] = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

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
        # Install the official SAM2.1 package from Meta AI
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"])
        print("SAM2 installed successfully")
    except Exception as e:
        print(f"Error installing SAM2: {e}")
        print("Please install manually: pip install git+https://github.com/facebookresearch/sam2.git")

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