#!/usr/bin/env python3
"""
Test script to verify installation of Intelligent Image Cropper dependencies.
"""

import sys
import importlib
import subprocess

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name or module_name}: {e}")
        return False

def test_torch():
    """Test PyTorch installation and CUDA availability."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available, will use CPU")

        return True
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False

def test_opencv():
    """Test OpenCV installation."""
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False

def test_ai_models():
    """Test AI model packages."""
    models = [
        ("groundingdino", "Grounding DINO"),
        ("segment_anything", "SAM2"),
        ("supervision", "Supervision"),
        ("transformers", "Transformers")
    ]

    results = []
    for module, name in models:
        results.append(test_import(module, name))

    return all(results)

def test_basic_functionality():
    """Test basic functionality with sample data."""
    try:
        import numpy as np
        import cv2

        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_image, (20, 20), (80, 80), (255, 0, 0), -1)

        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        print("✓ Basic image processing functionality works")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Testing Intelligent Image Cropper Installation")
    print("="*50)

    tests = [
        ("PyTorch", test_torch),
        ("OpenCV", test_opencv),
        ("AI Models", test_ai_models),
        ("Basic Functionality", test_basic_functionality)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        results.append(test_func())

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation is complete.")
        print("\nYou can now:")
        print("1. Download models: python download_models.py")
        print("2. Run demo: python demo.py")
        print("3. Process images: python crop_images.py --input_dir ./images --output_dir ./cropped --include 'person' --exclude 'background'")
    else:
        print("⚠ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Run: python setup.py")
        print("2. Install missing packages manually")
        print("3. Check Python version (requires 3.11+)")

if __name__ == "__main__":
    main()