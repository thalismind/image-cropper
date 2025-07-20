# Project Structure

This document describes the structure of the Intelligent Image Cropper project.

## Directory Structure

```
image-cropper/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Installation script
├── download_models.py          # Model download script
├── test_installation.py        # Installation verification
├── crop_images.py              # Main processing script
├── demo.py                     # Demo script
├── example_usage.py            # Example usage scenarios
├── PROJECT_STRUCTURE.md        # This file
└── models/                     # AI model modules
    ├── __init__.py
    ├── grounding_dino_detector.py  # Grounding DINO detection
    ├── sam2_segmenter.py           # SAM2 segmentation
    └── intelligent_cropper.py      # Main cropping logic
```

## Core Components

### 1. Model Modules (`models/`)

#### `grounding_dino_detector.py`
- **Purpose**: Text-guided object detection using Grounding DINO
- **Key Class**: `GroundingDINODetector`
- **Features**:
  - Loads and manages Grounding DINO model
  - Detects objects based on text prompts
  - Supports multiple class detection
  - Returns bounding boxes and confidence scores

#### `sam2_segmenter.py`
- **Purpose**: Precise object segmentation using SAM2.1
- **Key Class**: `SAM2Segmenter`
- **Features**:
  - Loads and manages SAM2.1 model
  - Creates precise masks from bounding boxes
  - Combines multiple segmentations
  - Provides visualization capabilities

#### `intelligent_cropper.py`
- **Purpose**: Main cropping logic that combines detection and segmentation
- **Key Class**: `IntelligentCropper`
- **Features**:
  - Orchestrates detection and segmentation
  - Finds optimal crop areas
  - Maximizes included areas while excluding unwanted regions
  - Provides visualization and crop area adjustment

### 2. Main Scripts

#### `crop_images.py`
- **Purpose**: Main command-line interface for batch processing
- **Features**:
  - Command-line argument parsing
  - Batch image processing
  - Progress tracking with tqdm
  - Comprehensive error handling
  - Support for glob patterns

#### `demo.py`
- **Purpose**: Demonstration script with sample images
- **Features**:
  - Creates sample images for testing
  - Runs multiple example configurations
  - Interactive mode for user input
  - Visualizes results

#### `example_usage.py`
- **Purpose**: Comprehensive examples and usage patterns
- **Features**:
  - Multiple real-world scenarios
  - Command-line usage examples
  - Different cropping strategies

### 3. Setup and Installation

#### `setup.py`
- **Purpose**: Automated installation script
- **Features**:
  - Installs all required packages
  - Creates necessary directories
  - Provides installation guidance

#### `download_models.py`
- **Purpose**: Downloads AI model files
- **Features**:
  - Downloads Grounding DINO models
  - Downloads SAM2 models
  - Handles download errors gracefully

#### `test_installation.py`
- **Purpose**: Verifies installation completeness
- **Features**:
  - Tests all package imports
  - Checks CUDA availability
  - Validates basic functionality

## Data Flow

1. **Input**: Images are loaded from input directory
2. **Detection**: Grounding DINO detects objects based on text prompts
3. **Segmentation**: SAM2.1 creates precise masks for detected objects
4. **Analysis**: System analyzes which areas to include/exclude
5. **Cropping**: Optimal crop area is calculated and applied
6. **Output**: Cropped images are saved to output directory

## Configuration Options

### Detection Parameters
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `min_area`: Minimum area for detected objects (pixels)
- `padding`: Padding around crop area (pixels)

### Processing Options
- `include_classes`: List of classes to include in crop
- `exclude_classes`: List of classes to exclude from crop
- `glob_pattern`: File pattern for image selection
- `save_visualizations`: Save detection/segmentation visualizations

## Model Files

The system requires the following model files (downloaded automatically):

### Grounding DINO
- `models/groundingdino/groundingdino_swint_ogc.py` (config)
- `models/groundingdino/groundingdino_swint_ogc.pth` (weights)

### SAM2.1
- `models/sam2/sam2.1_hiera_tiny.pt` (SAM2.1 Tiny model)
- `models/sam2/sam2.1_hiera_small.pt` (SAM2.1 Small model)
- `models/sam2/sam2.1_hiera_base_plus.pt` (SAM2.1 Base Plus model)
- `models/sam2/sam2.1_hiera_large.pt` (SAM2.1 Large model)

## Output Structure

```
output_directory/
├── cropped_image1.jpg          # Cropped images
├── cropped_image2.jpg
├── vis_cropped_image1.jpg      # Visualizations (if enabled)
└── vis_cropped_image2.jpg
```

## Error Handling

The system includes comprehensive error handling for:
- Missing model files
- Invalid image files
- Network issues during model download
- CUDA/GPU memory issues
- Invalid crop areas
- File I/O errors

## Performance Considerations

- **GPU Usage**: CUDA-compatible GPU recommended for optimal performance
- **Memory**: 8GB+ RAM recommended for large images
- **Batch Processing**: Process images in batches for efficiency
- **Model Loading**: Models are loaded once and reused for multiple images