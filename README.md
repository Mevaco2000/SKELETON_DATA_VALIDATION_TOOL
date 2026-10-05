# Validation Toolkit for Keypoint Datasets

This project provides tools for:

- keypoint quality validation,
- mislabeled sample detection,
- merged ranking based on multiple unsupervised signals,
- YOLO Pose workflows,
- format conversion (including COCO keypoints),
- an interactive Streamlit demo.

Core modules are under `utils/`:

- `utils/validation` for image- and dataset-level validation,
- `utils/datasets` for dataset operations, formats, and converters.

## 1. Requirements

- Python 3.10+
- Windows/Linux/macOS
- Optional GPU for faster model execution

## 2. Installation

Clone and install (Windows PowerShell):

```powershell
git clone https://github.com/Mevaco2000/YOLO_POSE_Data_Validation.git
cd YOLO_POSE_Data_Validation

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you downloaded a ZIP instead of cloning, extract it and run the same commands from the project root directory (the folder that contains `README.md`, `requirements.txt`, and `utils/`).

Notes:

- Conda environments are also supported.
- If you need a specific PyTorch build (for a CUDA version), install that `torch` build first, then run `pip install -r requirements.txt`.

## 3. Post-install check

Run smoke tests:

```powershell
python test_validation_smoke.py
```

Expected final line:

```text
All smoke tests passed.
```

## 4. Quick API usage

```python
from utils import YOLOPoseDataset, YPSetValidation, MergedRankingEvaluator

dataset = YOLOPoseDataset("path/to/train.txt")
validator = YPSetValidation(dataset)

evaluator = MergedRankingEvaluator(
    validator=validator,
    report_paths={},
    top_k=200,
    output_path="ranking_report.txt",
    distance_model_name="random_forest",
    segmentation_model_name=None,
)

results = evaluator.run(
    distance_weight=1.0,
    lbp_weight=1.0,
    segmentation_weight=0.0,
)

print("Ranking rows:", len(results.get("combined_ranking", [])))
```

## 5. Working with multiple input formats

Registered formats include:

- `yolo_pose`
- `coco_keypoints`

Example using the adapter:

```python
from utils.datasets import KeypointDatasetAdapter

with KeypointDatasetAdapter.from_coco(
    annotations_dir="path/to/coco/annotations",
    images_dir="path/to/coco/images",
    output_root="./converted_yolo_pose",
    split_name="train",
) as adapter:
    dataset = adapter.as_yolo_dataset()
    print("Samples:", len(dataset))
    print(adapter.info())
```

## 6. Streamlit demo

Run:

```powershell
streamlit run streamlit_merged_ranking_demo.py
```

The UI is fully in English and supports different input formats.

In the sidebar, configure:

- `Input dataset format` (for example `yolo_pose` or `coco_keypoints`),
- `Dataset path or URL` (directory/file/URL: `.zip`, `.yaml`, `.yml`, `.txt`, `.json`),
- optional COCO conversion settings,
- ranking weights and model options.

For non-YOLO formats, the app converts the dataset internally to YOLO Pose before ranking.

Results are written to:

- `streamlit_outputs/run_YYYYMMDD_HHMMSS/`

Including:

- `combined_ranking.csv`
- `top_k_ranked_paths.txt`
- `annotated_top_k/`
- `annotated_manifest.csv`

### Streamlit Cloud deployment notes

This repository includes two deployment helper files:

- `runtime.txt` (pins Python to 3.11)
- `packages.txt` (installs Linux system libs required by OpenCV)

If you deploy on Streamlit Cloud and see an error like `ImportError: libGL.so.1`, redeploy after making sure these files are in the repository root.
