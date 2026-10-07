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
git clone https://github.com/Mevaco2000/SKELETON_DATA_VALIDATION_TOOL.git
cd SKELETON_DATA_VALIDATION_TOOL
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation with `running scripts is disabled on this system`, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This change affects only the current PowerShell session.

If you downloaded a ZIP instead of cloning, extract it and run the same commands from the project root directory (the folder that contains `README.md`, `requirements.txt`, and `utils/`).

Notes:


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

Results are written to:

- `streamlit_outputs/run_YYYYMMDD_HHMMSS/`


- `combined_ranking.csv`
- `top_k_ranked_paths.txt`

This repository includes two deployment helper files:

- `runtime.txt` (pins Python to 3.11)
- `packages.txt` (installs Linux system libs required by OpenCV)

If your dataset ZIP is large (for example ~150 MB or more), make sure `.streamlit/config.toml` is present in the repository root and redeploy so Streamlit applies:

- `server.maxUploadSize = 1024`
- `server.maxMessageSize = 1024`

Run the full app stack with Docker Compose:

```powershell
docker compose up --build
```

Then open:

- `http://localhost:8501`

Stop the app:

```powershell
docker compose down
```

Notes:

- `streamlit_outputs/` is mounted as a volume, so generated outputs persist outside the container.
- `.streamlit/config.toml` is mounted too, so upload limits stay configurable from the repo.
