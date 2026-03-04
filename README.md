# Utils Package Documentation

## Overview

The **Utils** package is a professional toolkit for YOLO pose dataset validation, analysis, and manipulation. It provides utilities for detecting duplicate images, analyzing keypoint annotations, training and evaluating YOLO pose models, and performing dataset operations (flattening, merging, splitting, converting formats).

**Version:** 0.1.0  
**Author:** Rafał Wysocki

---

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Core Modules](#core-modules)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```python
   from utils import load_keypoints, train_yolo_pose_model
   print("✓ Utils package ready")
   ```

---

## Package Structure

```
utils/
├── __init__.py                  # Main package exports
├── config.py                    # Constants & configuration
│
├── validation/                  # Image validation & analysis
│   ├── __init__.py
│   ├── helpers.py              # Keypoint utilities
│   ├── image_validation.py     # Per-image LBP and feature extraction (YPImageValidation)
│   ├── set_validation.py       # Dataset-level analysis (YPSetValidation)
│   └── analysis.py             # Group analysis & Excel export
│
└── datasets/                    # Dataset manipulation
    ├── __init__.py
    ├── yolo_pose_dataset.py    # YOLO dataset handling (YOLOPoseImage, YOLOPoseDataset)
    └── operations.py           # Dataset operations (flatten, merge, split, convert)
```

---

## Core Modules

### 1. **Validation Module** (`utils.validation`)

Handles image validation, LBP feature extraction, and analysis.

#### Main Classes:

**YPImageValidation** — Per-image LBP and feature extraction
- `compute_lbp_histograms(patch_size)` — LBP histograms for all keypoints in image
- `compute_lbp_decimals(patch_size)` — Binary decimal representation for keypoints
- `compute_lbp_histogram_single(x, y, patch_size)` — Single keypoint LBP histogram
- `compute_lbp_decimal_single(x, y, patch_size)` — Single keypoint decimal representation
- `sequential_distances(visibility_threshold)` — Distances between consecutive keypoints
- `visualize(patch_size)` — Visualize keypoints with LBP patches

**YPSetValidation** — Dataset-level operations
- `from_yolo_dataset(dataset, patch_size)` — Create from YOLOPoseDataset
- `get_all_lbp_histograms()` — Get all LBP histograms for dataset
- `get_all_lbp_decimals()` — Get all binary decimals for dataset
- `create_dataframe()` — Export features to pandas DataFrame

**Analysis Functions** — Group visualization and reporting
- `generate_group_visualizations()` — Create visual group reports
- `save_groups_analysis()` — Save group data to JSON
- `analyze_hidden_keypoints()` — Analyze hidden/invalid keypoints
- `export_groups_analysis_to_excel()` — Export statistics to Excel

**helpers.py** — Keypoint utilities
- `load_keypoints()` — Load keypoints from YOLO label files
- `draw_keypoints()` — Draw keypoints on images

**analysis.py** — Group visualization and reporting
- `generate_group_visualizations()` — Create visual group reports
- `save_groups_analysis()` — Save group data to JSON
- `analyze_hidden_keypoints()` — Analyze hidden/invalid keypoints
- `export_groups_analysis_to_excel()` — Export statistics to Excel

---

#### Alternative: YPSetValidation for Dataset-Level Operations

For full dataset analysis, use `YPSetValidation`:

```python
from utils.validation import YPSetValidation, YOLOPoseDataset

# Create from YOLOPoseDataset
dataset = YOLOPoseDataset("data.yaml")
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)

# Get all LBP features
histograms = validator.get_all_lbp_histograms()  # (total_keypoints, 256)
decimals = validator.get_all_lbp_decimals()      # (total_keypoints,)

# Export to DataFrame
df = validator.create_dataframe()
df.to_csv("features.csv")
```

---

### 2. **Datasets Module** (`utils.datasets`)

Handles dataset manipulations and format conversions.

#### Functions in operations.py:

- `flatten_cvat_yolo_pose()` — Flatten nested CVAT dataset structure
- `merge_yolo_pose_datasets()` — Merge two YOLO datasets
- `split_yolo_pose_dataset()` — Split dataset into train/validation
- `convert_yolo_pose_to_cvat()` — Convert YOLO format to CVAT ZIP archives
- `extract_image_subset()` — Extract images by index range

---

### 3. **Configuration** (`utils.config`)

Central place for constants used across the package.

Key constants:
- `MODEL_REGISTRY` — Available YOLO model paths
- `VALID_IMAGE_EXTENSIONS` — Supported image formats
- `CLIP_MODEL_NAME` — CLIP model identifier
- `DEFAULT_THRESHOLD_SIMILARITY` — CLIP similarity threshold (default: 0.98)
- `DEFAULT_THRESHOLD_DUPLICATES` — Duplicate detection threshold (default: 0.99)
- `DEFAULT_K_NEIGHBORS` — FAISS search neighbors (default: 10)
- `DEFAULT_YOLO_EPOCHS` — Training epochs (default: 100)
- `DEFAULT_YOLO_IMGSZ` — Training image size (default: 640)
- `DEFAULT_YOLO_BATCH` — Training batch size (default: 16)

Modify `utils/config.py` to change defaults project-wide.

---

## Configuration

### Model Registry

Models are registered in `utils/config.py`:

```python
MODEL_REGISTRY = {
    "custom": None,  # path will be provided dynamically
}
```

Provide custom models directly to `train_yolo_pose_model()`.

### CLIP Settings

Adjust CLIP embedding parameters in `config.py`:

```python
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_MODEL_PRETRAINED = "openai"
```

### Thresholds

Tune detection thresholds for your use case:

```python
DEFAULT_THRESHOLD_SIMILARITY = 0.98    # Group similar images
DEFAULT_THRESHOLD_DUPLICATES = 0.99    # Find near-duplicates
```

---

## NEW: LBP Feature Extraction

### YOLOPoseDataset Class

Unified interface for loading and processing YOLO Pose datasets with optional LBP computation.
---

## NEW: LBP Feature Extraction

### YPImageValidation - Per-Image LBP Feature Extraction

The `YPImageValidation` class provides powerful LBP (Local Binary Pattern) feature extraction for individual images.

```python
from utils.validation import YPImageValidation
import cv2
import numpy as np

# Create validator for single image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
keypoints = np.array([[100, 50], [150, 200]])  # (N, 2) with [x, y]

validator = YPImageValidation(image, keypoints)

# Get LBP histograms for all keypoints
histograms = validator.compute_lbp_histograms(patch_size=5)  # (N, 256)

# Get binary decimal representation
decimals = validator.compute_lbp_decimals(patch_size=7)  # (N,)

# Compute for single keypoint
single_hist = validator.compute_lbp_histogram_single(100, 50, patch_size=5)  # (256,)

# Calculate distances between consecutive keypoints
distances = validator.sequential_distances(visibility_threshold=0.5)

# Visualize results
visualized = validator.visualize(patch_size=5)
cv2.imshow("Visualization", visualized)
```

**Key Methods:**
- `compute_lbp_histograms(patch_size)` — LBP histograms for all keypoints (N, 256)
- `compute_lbp_decimals(patch_size)` — Binary decimal representation (N,)
- `compute_lbp_histogram_single(x, y, patch_size)` — Single keypoint histogram (256,)
- `compute_lbp_decimal_single(x, y, patch_size)` — Single keypoint decimal
- `sequential_distances(visibility_threshold)` — Distance between consecutive keypoints
- `visualize(patch_size)` — Draw keypoints with LBP patches

### YPSetValidation - Dataset-Level Operations

For processing entire datasets efficiently:

```python
from utils.validation import YPSetValidation
from utils.datasets import YOLOPoseDataset

# Load dataset
dataset = YOLOPoseDataset("data.yaml")

# Create validator
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)

# Get all histograms at once
histograms = validator.get_all_lbp_histograms()  # (total_keypoints, 256)

# Get all decimals
decimals = validator.get_all_lbp_decimals()      # (total_keypoints,)

# Export to pandas DataFrame
df = validator.create_dataframe()
print(df.head())
df.to_csv("lbp_features.csv")
```


---

## API Reference

### Validation Module - YPImageValidation

```python
validator = YPImageValidation(image, keypoints)

# LBP Feature Extraction
histograms = validator.compute_lbp_histograms(patch_size=5)       # (N, 256)
decimals = validator.compute_lbp_decimals(patch_size=7)           # (N,)
single_hist = validator.compute_lbp_histogram_single(x, y)        # (256,)
single_dec = validator.compute_lbp_decimal_single(x, y)           # scalar

# Distance Analysis
distances = validator.sequential_distances(visibility_threshold=0.5)  # (N-1,)

# Visualization
result = validator.visualize(patch_size=5)
```

### Validation Module - YPSetValidation

```python
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)

# Get all features
histograms = validator.get_all_lbp_histograms()  # (total_keypoints, 256)
decimals = validator.get_all_lbp_decimals()      # (total_keypoints,)

# Export
df = validator.create_dataframe()
df.to_csv("features.csv")
```

### Validation Module - Analysis Functions

```python
from utils.validation import (
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel
)

# Visualize groups
generate_group_visualizations(groups, filenames, image_folder, label_folder, output_folder)

# Save analysis
save_groups_analysis(groups, filenames, label_folder, output_json)

# Analyze visibility
analyze_hidden_keypoints(label_folder, output_json)

# Export report
export_groups_analysis_to_excel(json_path, output_path)
```

### Datasets Module - YOLOPoseDataset

```python
dataset = YOLOPoseDataset("data.yaml")

# Properties
len(dataset)                      # Total images
dataset.get_stats()               # Dictionary of stats
dataset.get_image_paths()         # List of paths
dataset.get_label_paths()         # List of label paths

# Iteration
for sample in dataset:
    image = sample['image']       # numpy array
    keypoints = sample['keypoints']  # (N, 2) or (N, 3)
    image_path = sample['image_path']
    label_path = sample['label_path']

# Direct access
sample = dataset[0]
```

### Datasets Module - YOLOPoseImage

```python
from utils.datasets import YOLOPoseImage

img = YOLOPoseImage("image.jpg", "image.txt")

# Access data
print(img.image)         # numpy array
print(img.keypoints)     # (N, 2) or (N, 3)
print(img.image_path)
print(img.label_path)
```

### Datasets Module - Operations

```python
from utils.datasets import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset
)

# Flatten CVAT structure
flatten_cvat_yolo_pose(input_root, output_root)

# Merge datasets
merge_yolo_pose_datasets(dataset1_root, dataset2_root, output_root)

# Split into train/validation
split_yolo_pose_dataset(dataset_root, output_root, val_ratio=0.2, seed=42)

# Convert format
convert_yolo_pose_to_cvat(dataset_root, output_zip_dir)

# Extract subset
extract_image_subset(images_path, output_dir, a=0, b=100)
```

### Helpers Module

```python
from utils.validation import YPImageValidation

# Load keypoints from file
keypoints = YPImageValidation.load_keypoints("labels/image.txt")  # (N, 2)

# Draw keypoints on image
marked = YPImageValidation.draw_keypoints_on_image(image, keypoints, color=(0, 255, 0))

# Calculate sequential distances
distances = YPImageValidation.compute_sequential_distances(keypoints)  # (N-1,)
```

---

## Important Notes

### Type Hints

All functions include full type hints for better IDE support and type checking:
```bash
mypy utils/  # Run type checker
```

### Error Handling

The package validates input paths and raises informative errors:
- `FileNotFoundError` — Missing files or directories
- `IndexError` — Out-of-bounds parameters
- `ValueError` — Invalid parameter values

### Memory Efficiency

For large datasets, use iterators:
```python
# Good - memory efficient
for sample in dataset:
    process(sample)

# Avoid - loads all data at once
data = [sample for sample in dataset]
```

### Visibility Scores

Some keypoint arrays include visibility scores (N, 3) with [x, y, visibility]:
```python
keypoints = np.array([[100, 50, 1.0], [150, 200, 0.0]])  # (N, 3)
distances = validator.sequential_distances(visibility_threshold=0.5)
```

---

## Configuration

All settings can be optionally customized in `utils/config.py`:

```python
from utils.config import (
    MODEL_REGISTRY,
    VALID_IMAGE_EXTENSIONS,
    DEFAULT_THRESHOLD_SIMILARITY,
    DEFAULT_THRESHOLD_DUPLICATES
)

# View defaults
print(VALID_IMAGE_EXTENSIONS)  # ['.jpg', '.png', ...]
```

---

## Troubleshooting

**Q: Import errors with utils**  
A: Ensure requirements are installed and you're using correct paths:
```bash
pip install -r requirements.txt
```

**Q: Dataset loading fails**  
A: Verify data.yaml exists and paths are correct:
```python
import os
assert os.path.exists("data.yaml")
dataset = YOLOPoseDataset("data.yaml")
```

**Q: LBP feature extraction is slow**  
A: Use `YPSetValidation` for batch operations (faster than per-image):
```python
# Fast
validator = YPSetValidation.from_yolo_dataset(dataset)
features = validator.get_all_lbp_histograms()

# Slow (don't do this)
for sample in dataset:
    validator = YPImageValidation(sample['image'], sample['keypoints'])
    features = validator.compute_lbp_histograms()
```

**Q: Out of memory with large datasets**  
A: Use iteration instead of loading all at once:
```python
# Good
for sample in dataset:
    process(sample)

# Bad
samples = list(dataset)  # Loads everything into memory
```

### Device Management

GPU acceleration is automatic:
```python
# Uses GPU if available, falls back to CPU
embeddings, filenames = generate_clip_embeddings("./images")

# Force CPU
embeddings, filenames = generate_clip_embeddings("./images", device="cpu")
```

### Progress Bars

Operations use `tqdm` for progress tracking with automatic ETA:
```
Generating embeddings: 45%|████▌| 450/1000 [02:30<03:00, 3.05it/s]
```

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use CPU:
```python
train_yolo_pose_model(..., device="cpu", batch=8)
```

### Issue: Slow embedding generation
**Solution:** Ensure you're using GPU:
```python
generate_clip_embeddings(image_folder, device="cuda")
```

### Issue: Module import errors
**Solution:** Install package in development mode:
```bash
pip install -e .
```

### Issue: Corrupted images skipped silently
**Solution:** All corrupted images are automatically skipped. Check console output for count.

### Issue: Out-of-bounds index error
**Solution:** Ensure index range is valid:
```python
# Check total images first
import os
n_images = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
extract_image_subset(path, output, a=0, b=n_images-1)  # Valid range
```

---

## Contributing

To add new features:

1. Add functions to appropriate module
2. Include full type hints and docstrings
3. Update `__init__.py` exports
4. Test with sample data
5. Update this documentation

---

## License

© 2026 Data Validation Team

---

## Support

For issues or questions, check the example scripts or review function docstrings:
```python
from utils import load_keypoints
help(load_keypoints)  # Show detailed documentation
```
