# API Cheat Sheet

Quick reference for all classes and functions in the `utils` package.

---

## YPImageValidation - Per-Image LBP Feature Extraction

```python
from utils.validation import YPImageValidation
import cv2
import numpy as np

# Initialize with image and keypoints
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
keypoints = np.array([[100, 50], [150, 200]])  # (N, 2)

validator = YPImageValidation(image, keypoints)

# Get LBP features for all keypoints
histograms = validator.compute_lbp_histograms(patch_size=5)      # (N, 256)
decimals = validator.compute_lbp_decimals(patch_size=7)          # (N,)

# Single keypoint LBP
single_hist = validator.compute_lbp_histogram_single(100, 50)    # (256,)
single_decimal = validator.compute_lbp_decimal_single(100, 50)   # scalar

# Distance between keypoints
distances = validator.sequential_distances(visibility_threshold=0.5)  # (N-1,)

# Visualize
result = validator.visualize(patch_size=5)
```

**All Methods:**
- `__init__(image, keypoints)`
- `compute_lbp_histograms(patch_size)` → (N, 256)
- `compute_lbp_decimals(patch_size)` → (N,)
- `compute_lbp_histogram_single(x, y, patch_size)` → (256,)
- `compute_lbp_decimal_single(x, y, patch_size)` → scalar
- `sequential_distances(visibility_threshold)` → (N-1,)
- `visualize(patch_size)` → numpy array

---

## YOLOPoseDataset - Dataset Management

```python
from utils.datasets import YOLOPoseDataset

# Load dataset from data.yaml, train.txt, or args.yaml
dataset = YOLOPoseDataset("data.yaml")

# Iterate through samples
for sample in dataset:
    image = sample['image']          # numpy array
    keypoints = sample['keypoints']  # (N, 2) or (N, 3)
    image_path = sample['image_path']
    label_path = sample['label_path']

# Get specific sample
sample = dataset[0]

# Get dataset info
print(len(dataset))           # Total images
print(dataset.get_stats())    # Statistics
print(dataset.get_image_paths())   # All image paths
```

**Constructor:**
- `YOLOPoseDataset(dataset_path, include_visibility=True)`
  - dataset_path: Path to data.yaml, train.txt, or args.yaml

**Methods:**
- `__len__()` → Total number of images
- `__getitem__(idx)` → Get single sample
- `__iter__()` → Iterate through samples
- `get_stats()` → Dataset statistics
- `get_image_paths()` → All image paths
- `get_label_paths()` → All label paths

---

## YOLOPoseImage - Single Image

```python
from utils.datasets import YOLOPoseImage

# Load single image with its label
yolo_image = YOLOPoseImage("path/to/image.jpg", "path/to/image.txt")

# Access data
print(yolo_image.image)         # numpy array
print(yolo_image.keypoints)     # (N, 2) or (N, 3)
print(yolo_image.image_path)
print(yolo_image.label_path)
```

---

## YPSetValidation - Dataset Feature Extraction

```python
from utils.validation import YPSetValidation
from utils.datasets import YOLOPoseDataset

# Create from YOLOPoseDataset
dataset = YOLOPoseDataset("data.yaml")
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)

# Extract all LBP features
histograms = validator.get_all_lbp_histograms()  # (total_keypoints, 256)
decimals = validator.get_all_lbp_decimals()      # (total_keypoints,)

# Export to DataFrame
df = validator.create_dataframe()
df.to_csv("features.csv")
```

**Methods:**
- `from_yolo_dataset(dataset, patch_size)` → YPSetValidation
- `get_all_lbp_histograms()` → (total_keypoints, 256)
- `get_all_lbp_decimals()` → (total_keypoints,)
- `create_dataframe()` → pandas.DataFrame

---

## Helpers - Keypoint Utilities

```python
from utils.validation import YPImageValidation
import cv2

# Load keypoints from YOLO label file
keypoints = YPImageValidation.load_keypoints("labels/image.txt")  # (N, 2)

# Draw keypoints on image
image = cv2.imread("image.jpg")
marked = YPImageValidation.draw_keypoints_on_image(image, keypoints, color=(0, 255, 0))
cv2.imwrite("marked.jpg", marked)

# Calculate distances
distances = YPImageValidation.compute_sequential_distances(keypoints)  # (N-1,)
```

**Static Functions:**
- `load_keypoints(label_path)` → (N, 2)
- `draw_keypoints_on_image(image, keypoints, color)` → marked image
- `compute_sequential_distances(keypoints, visibility_threshold)` → distances

---

## Analysis Module - Group Operations

```python
from utils.validation import (
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel
)

# Visualize groups
generate_group_visualizations(
    groups=[[0, 5, 10], [1, 6]],
    filenames=filenames,
    image_folder="./images",
    label_folder="./labels",
    output_folder="./visualizations"
)

# Save JSON analysis
save_groups_analysis(
    groups=groups,
    filenames=filenames,
    label_folder="./labels",
    output_json="analysis.json"
)

# Analyze hidden keypoints
analyze_hidden_keypoints(
    label_folder="./labels",
    output_json="hidden.json"
)

# Export Excel report
export_groups_analysis_to_excel(
    json_path="analysis.json",
    output_path="report.xlsx"
)
```

---

## Dataset Operations

```python
from utils.datasets import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset
)

# Flatten CVAT structure
flatten_cvat_yolo_pose(
    input_root="./raw",
    output_root="./flat"
)

# Merge datasets
merge_yolo_pose_datasets(
    dataset1_root="./dataset1",
    dataset2_root="./dataset2",
    output_root="./merged"
)

# Split into train/val
split_yolo_pose_dataset(
    dataset_root="./data",
    output_root="./split",
    val_ratio=0.2,
    seed=42
)

# Convert to CVAT
convert_yolo_pose_to_cvat(
    dataset_root="./data",
    output_zip_dir="./cvat_export"
)

# Extract image range
extract_image_subset(
    images_path="./images",
    output_dir="./subset",
    a=100,
    b=200
)
```

---

## Common Patterns

### Pattern 1: Load Dataset and Extract LBP Features

```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPSetValidation

dataset = YOLOPoseDataset("data.yaml")
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)
features = validator.get_all_lbp_histograms()
print(f"Shape: {features.shape}")  # (total_keypoints, 256)
```

### Pattern 2: Per-Image Analysis

```python
from utils.datasets import YOLOPoseImage
from utils.validation import YPImageValidation

img = YOLOPoseImage("image.jpg", "image.txt")
validator = YPImageValidation(img.image, img.keypoints)
hist = validator.compute_lbp_histograms(patch_size=5)
distances = validator.sequential_distances()
```

### Pattern 3: Batch Processing

```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPImageValidation

dataset = YOLOPoseDataset("data.yaml")
for sample in dataset:
    validator = YPImageValidation(sample['image'], sample['keypoints'])
    features = validator.compute_lbp_histograms(patch_size=5)
    # Process features...
```

---

## Configuration

```python
from utils.config import MODEL_REGISTRY, VALID_IMAGE_EXTENSIONS

# View registered models
print(MODEL_REGISTRY)

# Valid image formats
print(VALID_IMAGE_EXTENSIONS)

# Default thresholds
from utils.config import (
    DEFAULT_THRESHOLD_SIMILARITY,
    DEFAULT_THRESHOLD_DUPLICATES
)
```

## Validation Module

### Keypoint Utilities

```python
from utils import (
    load_keypoints,
    sequential_distances,
    draw_keypoints,
    train_distance_models,
    predict_distances,
    _distance_matrix_from_keypoints,
)

# Load keypoints from label file
kp = load_keypoints("label_001.txt")  # Returns: np.ndarray (N, 2)

# Get distances between consecutive keypoints
distances = sequential_distances(kp)  # Returns: np.ndarray (N-1,)

# build regression models that predict one distance from the others
# assume ``dataset`` is an iterable of keypoint arrays
# create distance matrix first
# dist_mat.shape == (num_examples, N-1)
dist_mat = _distance_matrix_from_keypoints(dataset)
models = train_distance_models(dist_mat, exclude_endpoints=True)

# predict on a single example
test_distances = dist_mat[0]
pred = predict_distances(models, test_distances)
print("predicted interior distances", pred)

# Draw keypoints on image
image = cv2.imread("image.jpg")
new_image = draw_keypoints(image, kp, color=(0, 255, 0))
```

### CLIP Embeddings & Duplicates

```python
from utils import generate_clip_embeddings, find_near_duplicates

# Generate embeddings (GPU accelerated)
embeddings, filenames = generate_clip_embeddings(
    image_folder="./images",
    device="cuda"  # or "cpu"
)

# Find duplicates
duplicates = find_near_duplicates(
    embeddings=embeddings,
    filenames=filenames,
    k=10,              # number of neighbors
    threshold=0.99     # similarity threshold
)
# Returns: List[(file1, file2, similarity), ...]
```

### Group Analysis

```python
from utils.validation import (
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel
)

# Visualize groups
generate_group_visualizations(
    groups=[[0, 5, 10], [1, 6]],  # List of index groups
    filenames=filenames,
    image_folder="./images",
    label_folder="./labels",
    output_folder="./output/viz"
)

# Save analysis to JSON
save_groups_analysis(
    groups=groups,
    filenames=filenames,
    label_folder="./labels",
    output_json="./groups.json"
)

# Analyze hidden keypoints
analyze_hidden_keypoints(
    label_folder="./labels",
    output_json="./hidden_keypoints.json"
)

# Export to Excel with statistics
export_groups_analysis_to_excel(
    json_path="./groups.json",
    output_path="./report.xlsx"
)
```

### YOLO Model Training & Evaluation

```python
from utils import (
    train_yolo_pose_model,
    load_yolo_pose_label,
    evaluate_model_on_dataset
)

# Train model
model = train_yolo_pose_model(
    model_key=\"custom\",                      # or provide path directly
    dataset_yaml_path=\"./data/data.yaml\",
    custom_model_path=None,               # Optional custom model path
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda",
    project="runs/pose",
    name="experiment_001"
)

# Load label as keypoints
kp = load_yolo_pose_label("./label.txt")  # Returns: np.ndarray

# Evaluate on validation set
report = evaluate_model_on_dataset(
    model_or_path="./best.pt",            # Path string or YOLO object
    images_dir="./test_images",
    labels_dir="./test_labels",
    imgsz=640,
    device="cuda"
)
# Returns: Dict with summary, worst_images, worst_keypoints_global
```

---

## Datasets Module

```python
from utils.datasets import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset
)

# Flatten nested CVAT structure
flatten_cvat_yolo_pose(
    input_root="./cvat_export/",
    output_root="./flat_data/",
    copy_data_yaml=True
)

# Merge two datasets
merge_yolo_pose_datasets(
    dataset1_root="./data1/",
    dataset2_root="./data2/",
    output_root="./merged/"
)

# Split into train/validation
split_yolo_pose_dataset(
    dataset_root="./data/",
    output_root="./split/",
    val_ratio=0.2,        # 20% validation, 80% training
    seed=42               # For reproducibility
)

# Convert to CVAT format
convert_yolo_pose_to_cvat(
    dataset_root="./data/",
    output_zip_dir="./cvat_export/"
)

# Extract image subset by index range
extract_image_subset(
    images_path="./images/",
    output_dir="./subset/",
    a=1000,               # Start index (inclusive)
    b=2000                # End index (inclusive)
)
```

---

## Configuration

```python
from utils.config import (
    MODEL_REGISTRY,
    VALID_IMAGE_EXTENSIONS,
    DEFAULT_THRESHOLD_SIMILARITY,
    DEFAULT_THRESHOLD_DUPLICATES,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_YOLO_EPOCHS,
    DEFAULT_YOLO_IMGSZ,
    DEFAULT_YOLO_BATCH,
)

# Access custom model
print(MODEL_REGISTRY["custom"])  # None (provide path dynamically)

# Modify defaults
threshold = DEFAULT_THRESHOLD_DUPLICATES  # 0.99
```

---

## Common Patterns

### Pattern 1: End-to-End Duplicate Detection
```python
from utils import generate_clip_embeddings, find_near_duplicates

embeddings, files = generate_clip_embeddings("./images", device="cuda")
dups = find_near_duplicates(embeddings, files, threshold=0.99)
print(f"Found {len(dups)} duplicates")
```

### Pattern 2: Full Dataset Pipeline
```python
from utils.datasets import split_yolo_pose_dataset
from utils import train_yolo_pose_model, evaluate_model_on_dataset

# Split data
split_yolo_pose_dataset("./data", "./split", val_ratio=0.2)

# Train
model = train_yolo_pose_model(
    model_key=\"custom\",
    dataset_yaml_path="./split/data.yaml",
    epochs=50
)

# Evaluate
report = evaluate_model_on_dataset(
    model_or_path=model,
    images_dir="./split/images/val",
    labels_dir="./split/labels/val"
)
```

### Pattern 3: Keypoint Analysis
```python
from utils import load_keypoints, sequential_distances
import numpy as np

kp = load_keypoints("./label.txt")
distances = sequential_distances(kp)

print(f"Keypoints: {kp.shape}")  # (17, 2)
print(f"Distances: {distances.shape}")  # (16,)
print(f"Mean distance: {np.mean(distances):.4f}")
print(f"Max distance: {np.max(distances):.4f}")
```

---

## Error Handling

```python
from utils import train_yolo_pose_model

try:
    model = train_yolo_pose_model(
        model_key="invalid",
        dataset_yaml_path="./data.yaml"
    )
except ValueError as e:
    print(f"Invalid model: {e}")  # Model 'invalid' not in registry

except FileNotFoundError as e:
    print(f"File not found: {e}")  # Dataset or model doesn't exist
```

---

## Performance Tips

| Task | Tip |
|------|-----|
| Embeddings | Use `device="cuda"` for 10-100x speedup |
| Large datasets | Increase `k` in `find_near_duplicates()` gradually |
| Memory | Reduce `batch` size if OOM errors occur |
| Speed | Use `device="cuda"` for all GPU-accelerated functions |

---

## Function Signature Reference

```python
# Validation: Keypoints
load_keypoints(label_path: str) → np.ndarray
sequential_distances(keypoints: np.ndarray) → np.ndarray

train_distance_models(distance_matrix: np.ndarray,
                      model_factory: Callable = LinearRegression,
                      exclude_endpoints: bool = True) → List[sklearn.base.BaseEstimator]

predict_distances(models: Sequence[BaseEstimator],
                  distances: Union[np.ndarray, Sequence[np.ndarray]],
                  exclude_endpoints: bool = True) → np.ndarray
draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color: Tuple) → np.ndarray

# Validation: CLIP
generate_clip_embeddings(image_folder: str, device: Optional[str]) → Tuple[np.ndarray, List[str]]
find_near_duplicates(embeddings: np.ndarray, filenames: List[str], k: int, threshold: float) → List[Tuple]

# Validation: Analysis
generate_group_visualizations(groups: List, filenames: List, image_folder: str, label_folder: str, output_folder: str) → None
save_groups_analysis(groups: List, filenames: List, label_folder: str, output_json: str) → None
analyze_hidden_keypoints(label_folder: str, output_json: str) → None
export_groups_analysis_to_excel(json_path: str, output_path: str) → None

# Validation: Evaluation
train_yolo_pose_model(model_key: str, dataset_yaml_path: str, ...) → YOLO
load_yolo_pose_label(label_path: str) → np.ndarray
evaluate_model_on_dataset(model_or_path: Union[YOLO, str], images_dir: str, labels_dir: str, ...) → Dict

# Datasets
flatten_cvat_yolo_pose(input_root: str, output_root: str, copy_data_yaml: bool) → None
merge_yolo_pose_datasets(dataset1_root: str, dataset2_root: str, output_root: str) → None
split_yolo_pose_dataset(dataset_root: str, output_root: str, val_ratio: float, seed: int) → None
convert_yolo_pose_to_cvat(dataset_root: str, output_zip_dir: str) → None
extract_image_subset(images_path: str, output_dir: str, a: int, b: int) → None
```

---

## Need More Help?

- **Quick Start?** → See `QUICK_START.md`
- **Full Docs?** → See `README.md`
- **Project Layout?** → See `PROJECT_STRUCTURE.md`
- **Function Help?** → Run `help(function_name)`
- **Type Info?** → Run `mypy utils/`

---

Last updated: February 26, 2026
