# utils-yolo-validation

Zestaw narzędzi Pythona do walidacji, analizy i manipulacji datasetami YOLO Pose.

**Wersja:** 0.1.0 · **Python:** 3.8+ · **Autor:** Rafał Wysocki, PUT

Pełna dokumentacja klas i funkcji: [API.md](API.md)

---

## Spis treści

1. [Instalacja](#instalacja)
2. [Struktura pakietu](#struktura-pakietu)
3. [Szybki start](#szybki-start)

---

## Instalacja

### Wymagania

- Python 3.8 lub nowszy
- GPU z CUDA (opcjonalne, przyspiesza CLIP i FAISS)

### Krok 1 — pobranie repozytorium

```bash
git clone https://github.com/rwcki/utils-yolo-validation.git
cd utils-yolo-validation
```

### Krok 3 — wirtualne środowisko

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Krok 4 — instalacja pakietu

Instalacja edytowalna (zmiany w kodzie od razu widoczne bez reinstalacji):

```bash
pip install -e .
```

Alternatywnie, tylko zależności bez instalowania pakietu jako modułu:

```bash
pip install torch numpy pillow opencv-python matplotlib open-clip-torch faiss-cpu tqdm openpyxl ultralytics scikit-learn pandas
```

Opcjonalne — GPU (zamiennik `faiss-cpu`):

```bash
pip install faiss-gpu
```

### Krok 5 — weryfikacja

```python
import utils
print(utils.__version__)          # 0.1.0
from utils import YOLOPoseDataset, YPImageValidation, YPSetValidation
print("OK")
```

---

## Struktura pakietu

```
utils/
├── __init__.py              # Główne eksporty pakietu
├── config.py                # Stałe i konfiguracja (MODEL_REGISTRY itp.)
│
├── datasets/
│   ├── __init__.py
│   ├── yolo_pose_dataset.py # YOLOSample, YOLOPoseImage, YOLOPoseDataset
│   └── operations.py        # Funkcje i klasy wynikowe dla operacji na datasetach
│
└── validation/
    ├── __init__.py
    ├── image_validation.py  # YPImageValidation — walidacja pojedynczego obrazu
    ├── set_validation.py    # YPSetValidation — walidacja całego datasetu
    └── weryfikacje/
        ├── __init__.py
        └── yp_validation_test.py  # YPValidation_Test — gromadzenie wyników testów
```

---

## Szybki start

### Załadowanie datasetu

```python
from utils import YOLOPoseDataset

ds = YOLOPoseDataset("data.yaml")          # lub train.txt / args.yaml
print(len(ds))                             # liczba obrazów
print(ds.stats)                            # słownik ze statystykami

sample = ds[0]
print(sample.image_path)                   # ścieżka do obrazu
print(sample.keypoints.shape)              # (N_osób, K, 3)
```

### Obliczenie cech LBP dla obrazu

```python
from utils import YOLOPoseImage, YPImageValidation

img = YOLOPoseImage("path/to/image.jpg", "path/to/label.txt")
val = YPImageValidation(img.image, img.keypoints)

decimals  = val.compute_lbp_decimals(patch_size=5)            # (K,)
distances = val.sequential_distances(visibility_threshold=2.0) # (K-1,)
```

### Analiza anomalii na całym datasecie

```python
from utils import YOLOPoseDataset, YPSetValidation

ds  = YOLOPoseDataset("data.yaml")
sv  = YPSetValidation(ds)

# Odległości i modele do detekcji anomalii
models, valid_idx = sv.train_distance_models()
report = sv.predict_distance_anomalies(models, valid_idx, top_k=50)

# Anomalie LBP (Isolation Forest)
lbp_report = sv.predict_lbp_anomalies_by_embedding_groups()
```

### Operacje na datasetach

```python
from utils import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
)

flatten_cvat_yolo_pose("input/", "output/")
merge_yolo_pose_datasets("ds1/", "ds2/", "merged/")
split_yolo_pose_dataset("merged/", "split/", val_ratio=0.2)
convert_yolo_pose_to_cvat("split/", "cvat_zips/")
```

### Zastępowanie keypointów (augmentacja)

```python
from utils import YOLOPoseDataset, replace_angle_preserving_keypoints

ds = YOLOPoseDataset("data.yaml")
results = replace_angle_preserving_keypoints(
    ds,
    sample_count=100,
    max_keypoints_to_shift=3,
    max_endpoint_shift=0.05,
    output_dataset_path="augmented/",
)
for r in results:
    print(r.console_output)
```
