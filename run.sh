#!/bin/bash

# Intelligent Image Cropper - Run Script
# This script activates the virtual environment and runs the specified script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_help() {
    echo "Intelligent Image Cropper - Run Script"
    echo "====================================="
    echo ""
    echo "Usage: $0 <script> [arguments...]"
    echo ""
    echo "Available scripts:"
    echo "  crop_images.py    - Main image cropping script"
    echo "  demo.py           - Demo script"
    echo "  example_usage.py  - Example usage scenarios"
    echo "  test_installation.py - Test installation"
    echo "  download_models.py - Download AI models"
    echo "  test_new_crop_logic.py - Test new crop algorithm"
    echo ""
    echo "Examples:"
    echo "  $0 crop_images.py --input_dir ./images --output_dir ./cropped --include 'person'"
    echo "  $0 demo.py"
    echo "  $0 demo.py --interactive"
    echo "  $0 test_installation.py"
    echo ""
    echo "If no script is specified, shows this help message."
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found!"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./setup_venv.sh"
    echo ""
    echo "Or create the virtual environment manually:"
    echo "  virtualenv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if script is provided
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

# Get the script name
SCRIPT_NAME="$1"
shift  # Remove script name from arguments

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    print_error "Script '$SCRIPT_NAME' not found!"
    echo ""
    echo "Available scripts:"
    ls -1 *.py 2>/dev/null | grep -v __pycache__ || echo "No Python scripts found"
    exit 1
fi

# Check if script is executable
if [ ! -x "$SCRIPT_NAME" ]; then
    print_warning "Making script executable..."
    chmod +x "$SCRIPT_NAME"
fi

# Run the script with arguments
print_status "Running $SCRIPT_NAME with arguments: $*"
echo ""

# Execute the script
python "$SCRIPT_NAME" "$@"

# Check exit code
if [ $? -eq 0 ]; then
    print_status "Script completed successfully!"
else
    print_error "Script failed with exit code $?"
    exit 1
fi