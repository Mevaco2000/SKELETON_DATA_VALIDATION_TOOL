# Quick Start Guide

## 5-Minute Setup

### 1. Installation
```bash
cd d:\Politechnika\Deadlift_publikacja_nr_2\Skrypty
pip install -r requirements.txt
```

### 2. Basic Import
```python
from utils import load_keypoints, generate_clip_embeddings, train_yolo_pose_model
```

---

## Common Tasks

### Find Duplicate Images
```python
from utils import generate_clip_embeddings, find_near_duplicates

# Get embeddings
embeddings, filenames = generate_clip_embeddings("./images")

# Find duplicates
duplicates = find_near_duplicates(embeddings, filenames, threshold=0.99)
print(f"Found {len(duplicates)} duplicate pairs")
```

**Output:**
```
('image_001.jpg', 'image_105.jpg', 0.992)
('image_042.jpg', 'image_200.jpg', 0.995)
...
```

---

### Extract Image Subset
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
✔ Skopiowano 1001 obrazów do ./subset
```

---

### Split Dataset
```python
from utils.datasets import split_yolo_pose_dataset

split_yolo_pose_dataset(
    dataset_root="./data",
    output_root="./data_split",
    val_ratio=0.2  # 80% train, 20% validation
)
```

**Creates:**
- `data_split/images/train/`
- `data_split/images/val/`
- `data_split/labels/train/`
- `data_split/labels/val/`
- `data_split/train.txt`
- `data_split/val.txt`

---

## Extract LBP Features from Dataset

```python
from utils.validation import YOLOPoseDataset

# Load dataset from data.yaml (auto-detects train/val/test split)
dataset = YOLOPoseDataset("path/to/data.yaml")

# Get all LBP histograms in ONE COMMAND
histograms, counts, paths = dataset.get_all_lbp_histograms(patch_size=5)
print(f"Shape: {histograms.shape}")  # (total_keypoints, 256)

# Or get binary decimal representation
decimals, counts, paths = dataset.get_all_lbp_decimals(patch_size=7)

# Create pandas DataFrame
df = dataset.create_dataframe(patch_size=5)
df.to_csv("features.csv")
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
