# Intelligent Image Cropper

A Python project that uses Segment Anything v2 (SAM2) and Grounding DINO to intelligently crop images based on AI-detected areas. The system can include areas associated with specific keywords while excluding areas associated with exclusion keywords.

## Features

- **AI-Powered Detection**: Uses Grounding DINO to detect objects based on text prompts
- **Precise Segmentation**: Uses Segment Anything v2 for accurate object segmentation
- **Smart Cropping**: Maximizes crop area while excluding unwanted regions
- **Batch Processing**: Processes multiple images using glob patterns
- **Flexible Configuration**: Supports inclusion and exclusion keywords

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

## Installation

### Quick Start (Linux/macOS)

1. Clone this repository:
```bash
git clone <repository-url>
cd image-cropper
```

2. Run the setup script:
```bash
./setup_venv.sh
```

3. Download required model files:
```bash
./run.sh download_models.py
```

4. Test the installation:
```bash
./run.sh test_installation.py
```

### Quick Start (Windows)

1. Clone this repository:
```cmd
git clone <repository-url>
cd image-cropper
```

2. Run the setup script:
```cmd
setup_venv.bat
```

3. Download required model files:
```cmd
run.bat download_models.py
```

4. Test the installation:
```cmd
run.bat test_installation.py
```

### Manual Installation

If you prefer manual installation:

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install AI model packages:
```bash
pip install groundingdino-py segment-anything
```

3. Download models using the download script:
```bash
python download_models.py
```

## Usage

### Quick Demo

Run the demo to see the system in action:
```bash
python demo.py
```

Or run the interactive demo:
```bash
python demo.py --interactive
```

### Basic Usage

**Linux/macOS:**
```bash
./run.sh crop_images.py --input_dir ./images --output_dir ./cropped --include "person,car" --exclude "background,text"
```

**Windows:**
```cmd
run.bat crop_images.py --input_dir ./images --output_dir ./cropped --include "person,car" --exclude "background,text"
```

### Advanced Usage

**Linux/macOS:**
```bash
./run.sh crop_images.py \
    --input_dir ./images \
    --output_dir ./cropped \
    --include "person,car,animal" \
    --exclude "background,text,logo" \
    --confidence 0.5 \
    --min_area 1000 \
    --padding 50
```

**Windows:**
```cmd
run.bat crop_images.py --input_dir ./images --output_dir ./cropped --include "person,car,animal" --exclude "background,text,logo" --confidence 0.5 --min_area 1000 --padding 50
```

### Examples

Run comprehensive examples:

**Linux/macOS:**
```bash
./run.sh example_usage.py
```

**Windows:**
```cmd
run.bat example_usage.py
```

### Parameters

- `--input_dir`: Directory containing input images
- `--output_dir`: Directory to save cropped images
- `--include`: Comma-separated keywords for areas to include
- `--exclude`: Comma-separated keywords for areas to exclude
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--min_area`: Minimum area for detected objects (default: 1000)
- `--padding`: Padding around crop area in pixels (default: 50)
- `--min_crop_size`: Minimum size for individual crop areas (default: 200)
- `--glob_pattern`: Glob pattern for image files (default: "*.jpg,*.jpeg,*.png")

## How It Works

1. **Object Detection**: Grounding DINO detects objects based on inclusion/exclusion keywords
2. **Segmentation**: SAM2 creates precise masks for detected objects
3. **Area Analysis**: System analyzes which areas to include/exclude
4. **Smart Cropping Logic**:
   - If no exclusion areas found: Keep the whole image
   - If inclusion areas found: Create individual crops for each area with configurable minimum size
   - If exclusions overlap: Adjust crops to minimize excluded regions
5. **Output**: Saves cropped images with optimal composition

## Model Information

- **Grounding DINO**: Text-guided object detection
- **Segment Anything v2**: Zero-shot segmentation with improved performance
- **Supervision**: Utilities for computer vision tasks

## Examples

### Including People, Excluding Background
```bash
./run.sh crop_images.py --input_dir photos --include "person" --exclude "background"
```

### Including Multiple Objects
```bash
./run.sh crop_images.py --input_dir photos --include "car,person,animal" --exclude "text,logo"
```

### Product Photography
```bash
./run.sh crop_images.py --input_dir products --include "product" --exclude "logo,text" --confidence 0.6
```

### Street Photography
```bash
./run.sh crop_images.py --input_dir street --include "person" --exclude "car,text" --padding 100
```

### With Visualizations
```bash
./run.sh crop_images.py --input_dir images --include "person,car" --exclude "background" --save_visualizations
```

### Multiple Individual Crops
```bash
./run.sh crop_images.py --input_dir images --include "person,car" --exclude "background" --min_crop_size 300
```

## Troubleshooting

### Common Issues

1. **Model download fails**: Check your internet connection and try running `python download_models.py` again.

2. **CUDA out of memory**: Reduce batch size or use CPU by setting device to "cpu" in the model initialization.

3. **No detections found**: Try lowering the confidence threshold with `--confidence 0.3`.

4. **Import errors**: Run `python test_installation.py` to check which packages are missing.

### Performance Tips

- Use GPU for faster processing (CUDA-compatible GPU recommended)
- Adjust confidence threshold based on your use case
- Use appropriate padding values for your image types
- Consider using `--max_images` for testing with large datasets

## License

MIT License - see LICENSE file for details.