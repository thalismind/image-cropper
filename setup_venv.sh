#!/bin/bash

# Intelligent Image Cropper - Virtual Environment Setup Script
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on any error

echo "Intelligent Image Cropper - Virtual Environment Setup"
echo "=================================================="

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    echo "Please install Python $required_version+ and try again."
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Check if virtualenv is available
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    pip3 install virtualenv
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

virtualenv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install AI model packages
echo "Installing AI model packages..."
pip install groundingdino-py segment-anything

# Create necessary directories
echo "Creating directories..."
mkdir -p models/groundingdino models/sam2 demo_output test_output

# Download models (optional - can be done later)
echo ""
echo "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download models: python download_models.py"
echo "3. Test installation: python test_installation.py"
echo "4. Run demo: python demo.py"
echo ""
echo "Or use the run script: ./run.sh --help"