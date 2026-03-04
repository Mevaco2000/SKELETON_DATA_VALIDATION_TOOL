# Project Structure & Organization

## Overview

This is a professional Python package for YOLO pose dataset validation and manipulation, organized into logical modules with clear separation of concerns.

---

## Directory Layout

```
Skrypty/                          # Project root
├── README.md                      # Full documentation
├── QUICK_START.md                 # 5-minute setup guide
├── CHANGELOG.md                   # Version history
├── requirements.txt               # Pip dependencies
├── pyproject.toml                # Modern Python project metadata
├── setup.py                       # Legacy setup script
├── .gitignore                     # Git ignore patterns
│
├── utils/                         # MAIN PACKAGE (import from here)
│   ├── __init__.py               # Package root & main exports
│   ├── config.py                 # Global constants & config
│   │
│   ├── validation/               # Image validation & analysis
│   │   ├── __init__.py           # Module exports
│   │   ├── helpers.py            # Keypoint utilities (public)
│   │   ├── duplicates.py         # CLIP & FAISS (public functions, private helpers)
│   │   ├── analysis.py           # Group analysis & Excel (public)
│   │   └── evaluation.py         # YOLO training & evaluation (public)
│   │
│   └── datasets/                 # Dataset operations
│       ├── __init__.py           # Module exports
│       └── operations.py         # Dataset manipulation (public)
│
├── Open_MPII_v_2_1001_2000/      # Sample datasets
├── Open_MPII_v_2_2001_3000/
├── Open_MPII_v_2_3001_4000/
│
├── runs/                          # YOLO training outputs
│   └── pose/
├── group_visualizations/          # Generated group visualizations
├── grouped_duplicates/            # Duplicate image groups
│
├── korygowanie datasetu 13_02/    # Legacy processing folder
│
└── [Script files]
    ├── wyciaganie_indexow_zbior_zdjec.py     # Uses extract_image_subset
    ├── Deleting_duplicates.py
    ├── Grouping_images.py
    ├── Sklejanie_datasetow.ipynb             # Dataset merging notebook
    ├── Testy_jednostkowe.ipynb               # Unit tests notebook
    ├── Weryfikacja_czy_sa_annotacje.py
    └── [other processing scripts]
```

---

## Module Descriptions

### 🎯 `utils/` Package

The main Python package providing data validation and dataset utilities.

**Import from here:**
```python
from utils import load_keypoints, train_yolo_pose_model
from utils.validation import generate_clip_embeddings
from utils.datasets import split_yolo_pose_dataset
```

---

### 📝 `utils/config.py`

Centralized configuration and constants.

**Contains:**
- `MODEL_REGISTRY` — YOLO model paths
- Image format constants
- Default thresholds and parameters
- Training hyperparameter defaults

**Usage:**
```python
from utils.config import MODEL_REGISTRY, DEFAULT_THRESHOLD_DUPLICATES
print(MODEL_REGISTRY["custom"])  # custom: None (provide path dynamically)
```

---

### ✅ `utils/validation/`

Image validation, LBP feature extraction, and analysis.

| Submodule | Purpose | Main Classes/Functions |
|-----------|---------|-----------------|
| `helpers.py` | Keypoint utilities | `load_keypoints()`, `draw_keypoints()` |
| `image_validation.py` | Per-image LBP analysis | `YPImageValidation` class |
| `set_validation.py` | Dataset-level analysis | `YPSetValidation` class |
| `analysis.py` | Group analysis & reporting | `generate_group_visualizations()`, `save_groups_analysis()`, `export_groups_analysis_to_excel()` |
| `evaluation.py` | LBP feature extraction & dataset handling | `YOLOPoseDataset` class with `get_all_lbp_histograms()`, `get_all_lbp_decimals()`, `apply_function_to_all()` |

---

### 📦 `utils/datasets/`

Dataset manipulation and format conversion utilities.

| Function | Purpose |
|----------|---------|
| `flatten_cvat_yolo_pose()` | Flatten nested CVAT structures to flat directories |
| `merge_yolo_pose_datasets()` | Merge multiple datasets with conflict handling |
| `split_yolo_pose_dataset()` | Split into train/validation sets |
| `convert_yolo_pose_to_cvat()` | Convert YOLO format to CVAT ZIP archives |
| `extract_image_subset()` | Extract images by index range |

---

## File Organization Principles

### 1. **Logical Grouping by Function**

```python
# Image validation tasks → validation/
from utils.validation import generate_clip_embeddings, find_near_duplicates

# Dataset operations → datasets/
from utils.datasets import split_yolo_pose_dataset, flatten_cvat_yolo_pose
```

### 2. **Public vs Private Functions**

- **Public** (no prefix): Intended for external use, fully documented
  - `load_keypoints()`
  - `generate_clip_embeddings()`
  - `split_yolo_pose_dataset()`

- **Private** (`_` prefix): Internal use only, not exported
  - `_load_clip_model()` — Only used inside `generate_clip_embeddings()`
  - `_compute_embeddings()` — Only used inside `generate_clip_embeddings()`
  - `_build_similarity_groups()` — Only used inside FAISS search

### 3. **Module-Level `__init__.py` Files**

Each module has an `__init__.py` that:
- Imports and re-exports public functions
- Provides a clean, documented interface
- Prevents direct imports of private functions

```python
# Good: Import from module level
from utils.validation import generate_clip_embeddings

# Bad: Don't import private functions
from utils.validation.duplicates import _load_clip_model
```

### 4. **Centralized Configuration**

All magic numbers, thresholds, and paths in `config.py`:

```python
# Good: Easy to find and change globally
from utils.config import DEFAULT_THRESHOLD_DUPLICATES

# Bad: Scattered constants across multiple files
threshold = 0.99  # Where did this come from?
```

---

## Data Flow

### Duplicate Detection Pipeline
```
raw images/ 
    ↓
generate_clip_embeddings()  [GPU accelerated]
    ↓
embeddings (numpy array)
    ↓
find_near_duplicates()     [FAISS search]
    ↓
list of (file1, file2, similarity)
```

### Dataset Processing Pipeline
```
raw dataset (nested structure)
    ↓
flatten_cvat_yolo_pose()
    ↓
flat images/ + labels/
    ↓
split_yolo_pose_dataset()
    ↓
train/ + val/ splits
    ↓
train_yolo_pose_model()
    ↓
trained model
```

---

## Import Patterns

### ✅ Recommended Import Styles

```python
# Module-level imports (clean, discoverable)
from utils import load_keypoints
from utils.validation import generate_clip_embeddings
from utils.datasets import split_yolo_pose_dataset

# Namespace imports (for multiple functions from same module)
from utils.validation import (
    generate_clip_embeddings,
    find_near_duplicates,
)

# Global config
from utils.config import MODEL_REGISTRY
```

### ❌ Avoid These Patterns

```python
# Don't: Wildcard imports (pollutes namespace)
from utils.validation import *

# Don't: Private function imports
from utils.validation.duplicates import _load_clip_model

# Don't: Deep relative imports
from ..utils.validation.duplicates import find_near_duplicates
```

---

## Common Tasks Across Modules

### Task: Load Configuration
```python
from utils.config import MODEL_REGISTRY, DEFAULT_THRESHOLD_DUPLICATES

# Custom model path
model_path = "path/to/your/model.pt"
threshold = DEFAULT_THRESHOLD_DUPLICATES
```

### Task: Use Multiple Validation Functions
```python
from utils.validation import (
    load_keypoints,
    sequential_distances,
    generate_clip_embeddings,
    find_near_duplicates,
)
```

### Task: Dataset + Validation Pipeline
```python
from utils.datasets import split_yolo_pose_dataset
from utils.validation import train_yolo_pose_model

# First, split dataset
split_yolo_pose_dataset("raw_data", "split_data", val_ratio=0.2)

# Then, train model on split
model = train_yolo_pose_model(
    model_key=\"custom\",
    dataset_yaml_path="split_data/data.yaml"
)
```

---

## Type Hints & IDE Support

All functions have complete type hints:

```python
def find_near_duplicates(
    embeddings: np.ndarray,        # IDE shows: numpy array expected
    filenames: List[str],           # IDE shows: list of strings
    k: int = 10,                    # IDE shows: integer with default 10
    threshold: float = 0.99         # IDE shows: float between 0-1
) -> List[Tuple[str, str, float]]:  # IDE shows: returns list of tuples
```

Benefits:
- ✅ IDE autocomplete works perfectly
- ✅ Type checkers (`mypy`) can validate code
- ✅ Self-documenting parameters
- ✅ Jump to definition works reliably

---

## Version Management

### Current Structure

- `__version__` defined in `utils/__init__.py`
- Major.Minor.Patch format: `0.1.0`
- See `CHANGELOG.md` for release notes

### Package Metadata

- `pyproject.toml` — Modern Python packaging (PEP 517)
- `setup.py` — Legacy compatibility
- `setup.cfg` — Alternative configuration

---

## Testing Structure (Future)

```
tests/
├── test_validation.py
├── test_datasets.py
├── test_helpers.py
└── fixtures/
    ├── sample_images/
    └── sample_labels/
```

Run tests:
```bash
pytest tests/
pytest tests/ --cov=utils  # With coverage
```

---

## Documentation Structure

| File | Purpose |
|------|---------|
| `README.md` | Full API documentation & usage guide |
| `QUICK_START.md` | 5-minute setup & common patterns |
| `CHANGELOG.md` | Version history & planned features |
| `PROJECT_STRUCTURE.md` | This file — organization & principles |

---

## Design Philosophy

1. **Simplicity**: Functions do one thing well
2. **Clarity**: Type hints & docstrings everywhere
3. **Reusability**: Build complex operations from simple functions
4. **Testability**: Pure functions with clear inputs/outputs
5. **Documentation**: Every public function is documented

---

## Future Extensibility

Adding new features:

1. **New validation function?** → `utils/validation/new_module.py`
2. **New dataset operation?** → Add to `utils/datasets/operations.py`
3. **New constants?** → Add to `utils/config.py`
4. **Update exports** → Modify `__init__.py` files

---

Last updated: February 26, 2026
