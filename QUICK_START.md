# Quick Start Guide

## 5-Minute Setup

### 1. Installation
```bash
cd d:\Politechnika\Deadlift_publikacja_nr_2\Skrypty
pip install -r requirements.txt
```

### 2. Basic Import
```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPImageValidation, YPSetValidation
```

---

## Common Tasks

### Task 1: Load a Dataset

```python
from utils.datasets import YOLOPoseDataset

# Load from data.yaml
dataset = YOLOPoseDataset("path/to/data.yaml")

# Or from train.txt
dataset = YOLOPoseDataset("path/to/train.txt")

# Get info
print(f"Total images: {len(dataset)}")
print(f"Stats: {dataset.get_stats()}")

# Iterate through samples
for sample in dataset:
    image = sample['image']          # numpy array (H, W, 3)
    keypoints = sample['keypoints']  # (N, 2) or (N, 3)
    print(f"Image: {sample['image_path']}, Keypoints: {keypoints.shape}")
    
    # Stop after first 5
    if len([1 for _ in range(5)]) > 0:
        break
```

**Output:**
```
Total images: 1001
Stats: {'total_images': 1001, 'total_keypoints': 5005, ...}
Image: ./images/image_001.jpg, Keypoints: (5, 2)
```

---

### Task 2: Extract LBP Features from Single Image

```python
from utils.validation import YPImageValidation
from utils.datasets import YOLOPoseImage

# Load single image with keypoints
yolo_img = YOLOPoseImage("path/to/image.jpg", "path/to/image.txt")

# Create validator
validator = YPImageValidation(yolo_img.image, yolo_img.keypoints)

# Get LBP histograms for all keypoints
histograms = validator.compute_lbp_histograms(patch_size=5)
print(f"Histograms shape: {histograms.shape}")  # (N, 256)

# Get binary decimal representation
decimals = validator.compute_lbp_decimals(patch_size=7)
print(f"Decimals shape: {decimals.shape}")  # (N,)

# Calculate distances between keypoints
distances = validator.sequential_distances(visibility_threshold=0.5)
print(f"Distances: {distances}")
```

**Output:**
```
Histograms shape: (5, 256)
Decimals shape: (5,)
Distances: [23.5, 45.2, 12.1, 33.8]
```

---

### Task 3: Extract LBP Features from Entire Dataset

```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPSetValidation

# Load dataset
dataset = YOLOPoseDataset("path/to/data.yaml")

# Create validator for entire dataset
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)

# Get all LBP histograms in one command
histograms = validator.get_all_lbp_histograms()
print(f"Total histograms: {histograms.shape}")  # (total_keypoints, 256)

# Get all decimals
decimals = validator.get_all_lbp_decimals()
print(f"Total decimals: {decimals.shape}")  # (total_keypoints,)

# Export to DataFrame
df = validator.create_dataframe()
print(df.head())
df.to_csv("lbp_features.csv")
```

**Output:**
```
Total histograms: (5005, 256)
Total decimals: (5005,)
   image_idx  keypoint_idx  lbp_histogram  ... decimal
0          0              0       [...]     ... 12345
1          0              1       [...]     ... 23456
```

---

### Task 4: Split Dataset into Train/Validation

```python
from utils.datasets import split_yolo_pose_dataset

split_yolo_pose_dataset(
    dataset_root="./data",
    output_root="./data_split",
    val_ratio=0.2,  # 80% train, 20% validation
    seed=42
)
```

**Creates:**
```
data_split/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── train.txt
└── val.txt
```

---

### Task 5: Extract Subset of Images

```python
from utils.datasets import extract_image_subset

# Copy images 1000-2000
extract_image_subset(
    images_path="./full_dataset/images",
    output_dir="./subset",
    a=1000,
    b=2000
)
```

**Output:**
```
✔ Copied 1001 images to ./subset
```

---

### Task 6: Merge Two Datasets

```python
from utils.datasets import merge_yolo_pose_datasets

merge_yolo_pose_datasets(
    dataset1_root="./data1",
    dataset2_root="./data2",
    output_root="./data_merged"
)
```

**Output:**
```
data_merged/
├── images/
└── labels/

# Filenames are prefixed to avoid conflicts:
# dataset1_image_001.jpg
# dataset2_image_001.jpg  
```

---

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'utils'"**  
A: Make sure you're in the correct directory and requirements are installed:
```bash
pip install -r requirements.txt
```

**Q: "cv2 not found"**  
A: Install opencv-python:
```bash
pip install opencv-python
```

**Q: Dataset loader fails**  
A: Ensure your data.yaml path is correct and file exists:
```python
import os
print(os.path.exists("path/to/data.yaml"))
```

**Q: Memory issues with large datasets**  
A: Use iterator pattern instead of loading all at once:
```python
for sample in dataset:  # Loads one at a time
    # Process sample
```

**Output:**
```
✔ Loaded 127 images with 9 keypoints each
✔ Extracted LBP histograms: (1143, 256)
```

---
```python
from utils import train_yolo_pose_model

model = train_yolo_pose_model(
    model_key="custom",           # or provide path directly
    dataset_yaml_path="./data/data.yaml",
    epochs=50,
    batch=16,
    device="cuda"  # or "cpu"
)

print("✓ Model trained!")
```

---

### Evaluate Model
```python
from utils import evaluate_model_on_dataset

report = evaluate_model_on_dataset(
    model_or_path="./best.pt",
    images_dir="./test_images",
    labels_dir="./test_labels"
)

# View results
print(f"Mean error: {report['summary']['overall_mean_error']:.4f}")
print(f"Worst keypoints: {report['worst_keypoints_global'][:5]}")
```

---

### Load Keypoints
```python
from utils import load_keypoints, sequential_distances
import numpy as np

kp = load_keypoints("./label_001.txt")
print(f"Shape: {kp.shape}")  # (17, 2) for typical pose

# Calculate distances
distances = sequential_distances(kp)
print(f"Mean keypoint distance: {np.mean(distances):.4f}")
```

---

## Directory Structure

```
project/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── output/
│   ├── models/
│   ├── visualizations/
│   └── reports/
├── utils/              # Package (import from here)
│   ├── validation/
│   ├── datasets/
│   └── config.py
├── README.md
├── QUICK_START.md      # You are here
└── requirements.txt
```

---

## Examples

All example scripts that use these functions:

- `wyciaganie_indexow_zbior_zdjec.py` — Extract image subsets
- `Sklejanie_datasetow.ipynb` — Merge datasets
- `Testy_jednostkowe.ipynb` — Unit tests

---

## Tips

✅ **Use GPU** for 10-100x speedup:
```python
embeddings, filenames = generate_clip_embeddings(folder, device="cuda")
```

✅ **Adjust thresholds** for your dataset:
```python
# Stricter (fewer duplicates found)
find_near_duplicates(embeddings, filenames, threshold=0.999)

# Looser (more duplicates found)
find_near_duplicates(embeddings, filenames, threshold=0.95)
```

✅ **Check progress** — all operations show progress bars automatically

✅ **Skip corrupted images** — automatically skipped, no errors raised

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'utils'` | Run `pip install -r requirements.txt` |
| CUDA out of memory | Use CPU: `device="cpu"` or reduce batch size |
| Slow embeddings | Use GPU: `device="cuda"` |
| `FileNotFoundError` | Check path exists: `os.path.exists(path)` |
| Wrong image count | Verify image extensions: `.jpg`, `.png`, `.jpeg` |

---

## Next Steps

- Read [README.md](README.md) for full API documentation
- Check function docstrings: `help(load_keypoints)`
- Explore example notebooks in the project
- Modify `utils/config.py` to tune thresholds

---

Happy analyzing! 🚀
