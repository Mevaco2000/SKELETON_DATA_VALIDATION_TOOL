# API Reference

Dokumentacja wszystkich publicznych klas, metod i funkcji pakietu `utils`.

---

## Spis treści

- [utils.validation.image\_validation — YPImageValidation](#ypimagevalidation)
- [utils.validation.set\_validation — YPSetValidation](#ypsetvalidation)
- [utils.validation.weryfikacje — YPValidation\_Test](#ypvalidation_test)
- [utils.datasets.yolo\_pose\_dataset — YOLOPoseImage, YOLOPoseDataset](#yoloposedataset--yoloposeimage)
- [utils.datasets.operations — klasy wynikowe](#klasy-wynikowe-operacji)
- [utils.datasets.operations — funkcje](#funkcje-operacji-na-datasetach)

---

## YPImageValidation

`from utils.validation import YPImageValidation`  
`from utils import YPImageValidation`

Walidacja pojedynczego obrazu z adnotacjami keypointów. Oblicza cechy LBP, odległości i kąty między keypointami, ocenia keypointy względem maski segmentacji.

### Konstruktor

```python
YPImageValidation(
    image: Union[np.ndarray, str],
    keypoints: np.ndarray,
)
```

| Parametr | Typ | Opis |
|---|---|---|
| `image` | `np.ndarray` lub `str` | Tablica obrazu (H×W) lub ścieżka do pliku |
| `keypoints` | `np.ndarray` | Tablica shape `(K, 2)` lub `(K, 3)` — `[x, y]` lub `[x, y, vis]` |

### Metody instancji

#### `sequential_distances`
```python
.sequential_distances(
    visibility_threshold: float = 2.0,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray
```
Odległości euklidesowe między parami widocznych keypointów. Zwraca tablicę `(K-1,)` lub według `distance_connections`.

#### `sequential_angles`
```python
.sequential_angles(visibility_threshold: float = 2.0) -> np.ndarray
```
Kąty kolejnych odcinków keypoint-keypoint w radianach.

#### `get_sequential_distances`
```python
.get_sequential_distances(
    visibility_threshold: float = 2.0,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray
```
Alias dla `sequential_distances`.

#### `get_sequential_angles`
```python
.get_sequential_angles(visibility_threshold: float = 2.0) -> np.ndarray
```
Alias dla `sequential_angles`.

#### `evaluate_keypoints_against_mask`
```python
.evaluate_keypoints_against_mask(
    mask: np.ndarray,
    visibility_threshold: Optional[float] = 2.0,
) -> np.ndarray
```
Sprawdza, czy każdy keypoint leży wewnątrz maski binarnej. Zwraca tablicę `(K,)` boolowską / metryk.

#### `compute_lbp_decimal_single`
```python
.compute_lbp_decimal_single(x: int, y: int, patch_size: int = 5) -> int
```
Binarny zapis dziesiętny LBP dla pojedynczego keypointa (x, y).

#### `compute_lbp_decimals`
```python
.compute_lbp_decimals(
    patch_size: int = 5,
    visibility_threshold: float = 2.0,
) -> np.ndarray
```
Binarny zapis LBP dla wszystkich widocznych keypointów. Zwraca `(K,)`.

#### `draw_keypoints`
```python
.draw_keypoints(
    color: tuple = (0, 255, 0),
    keypoint_size: int = 4,
) -> np.ndarray
```
Rysuje keypointy jako okręgi na obrazie. Zwraca obraz BGR.

#### `draw_keypoints_with_patches`
```python
.draw_keypoints_with_patches(
    patch_size: int = 32,
    keypoint_color: tuple = (0, 255, 0),
    patch_color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray
```
Rysuje keypointy i prostokąty patchy wokół nich.

#### `visualize`
```python
.visualize(patch_size: int = 5)
```
Wyświetla obraz z keypointami i patchami (matplotlib).

### Metody statyczne

#### `compute_sequential_distances` *(static)*
```python
YPImageValidation.compute_sequential_distances(
    keypoints: np.ndarray,
    visibility_threshold: float = 2.0,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray
```

#### `compute_sequential_angles` *(static)*
```python
YPImageValidation.compute_sequential_angles(
    keypoints: np.ndarray,
    visibility_threshold: float = 2.0,
) -> np.ndarray
```

#### `compute_keypoint_mask_metrics` *(static)*
```python
YPImageValidation.compute_keypoint_mask_metrics(
    keypoints: np.ndarray,
    mask: np.ndarray,
    image_shape: tuple,
    visibility_threshold: Optional[float] = 2.0,
) -> np.ndarray
```
Wylicza przynależność do maski i odległości poziome od krawędzi maski.

#### `divide_distances_by_index` *(static)*
```python
YPImageValidation.divide_distances_by_index(distances: np.ndarray) -> dict
```
Grupuje tablicę odległości według indeksu odległości.

#### `distances_by_index_to_matrix` *(static)*
```python
YPImageValidation.distances_by_index_to_matrix(
    distances: np.ndarray,
    fill_value: float = np.nan,
) -> tuple
```
Konwertuje zgrupowane odległości do formatu macierzowego.

#### `draw_keypoints_on_image` *(static)*
```python
YPImageValidation.draw_keypoints_on_image(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: tuple = (0, 255, 0),
    keypoint_size: int = 4,
) -> np.ndarray
```

### Metody klasowe

#### `load_keypoints` *(classmethod)*
```python
YPImageValidation.load_keypoints(
    cls,
    label_path: str,
    include_visibility: bool = True,
    keypoint_format: str = 'auto',
) -> np.ndarray
```
Wczytuje keypointy pierwszej osoby z pliku etykiet YOLO Pose.

#### `load_all_keypoints_from_file` *(classmethod)*
```python
YPImageValidation.load_all_keypoints_from_file(
    cls,
    label_path: str,
    include_visibility: bool = True,
    keypoint_format: str = 'auto',
) -> List[np.ndarray]
```
Wczytuje keypointy wszystkich osób z pliku etykiet.

---

## YPSetValidation

`from utils.validation import YPSetValidation`  
`from utils import YPSetValidation`

Walidacja całego datasetu — oblicza cechy dla wszystkich obrazów, wykrywa duplikaty i anomalie.

### Konstruktor

```python
YPSetValidation(dataset)
```

| Parametr | Typ | Opis |
|---|---|---|
| `dataset` | `YOLOPoseDataset` | Załadowany dataset |

### Metody instancji

#### `get_embeddings`
```python
.get_embeddings(
    device: Optional[str] = None,
    image_folder: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]
```
Generuje embeddingi CLIP dla wszystkich obrazów. Zwraca `(embeddingi, ścieżki_obrazów)`.

#### `get_all_lbp_decimals`
```python
.get_all_lbp_decimals(
    patch_size: int = 5,
    verbose: bool = True,
    visibility_threshold: float = 2.0,
) -> tuple
```
LBP decimals dla wszystkich keypointów w datasecie.

#### `get_stats`
```python
.get_stats() -> dict
```
Statystyki datasetu (liczba obrazów, keypointów itd.).

#### `get_keypoint_statistics`
```python
.get_keypoint_statistics(verbose: bool = True) -> dict
```
Statystyki widoczności każdego keypointu (indeks → liczba widocznych).

#### `get_keypoint_distribution`
```python
.get_keypoint_distribution(verbose: bool = True) -> dict
```
Rozkład liczby widocznych keypointów na osobę.

#### `get_sequential_distances`
```python
.get_sequential_distances(
    verbose: bool = True,
    visibility_threshold: float = 2.0,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray
```
Odległości dla wszystkich osób we wszystkich obrazach.

#### `get_sequential_angles`
```python
.get_sequential_angles(
    verbose: bool = True,
    visibility_threshold: float = 2.0,
) -> np.ndarray
```
Kąty odcinków dla wszystkich osób.

#### `count_distances_by_index`
```python
.count_distances_by_index(
    distances_result: np.ndarray = None,
    verbose: bool = True,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> dict
```
Liczba odległości dla każdego indeksu odległości w datasecie.

#### `train_distance_models`
```python
.train_distance_models(
    model_factory: Union[Callable, str] = 'hist_gradient_boosting',
    exclude_endpoints: bool = False,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    visibility_threshold: float = 2.0,
    verbose: bool = True,
) -> Tuple[List[BaseEstimator], np.ndarray]
```
Trenuje modele regresji do predykcji jednej odległości z pozostałych. Zwraca `(modele, valid_distance_indices)`.

Dostępne wartości `model_factory`: `'hist_gradient_boosting'`, `'ridge'`, `'lasso'`, `'elastic_net'`, lub własna fabryka `Callable`.

#### `predict_distance_anomalies`
```python
.predict_distance_anomalies(
    models: Sequence[BaseEstimator],
    valid_distance_indices: np.ndarray,
    exclude_endpoints: bool = False,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    visibility_threshold: float = 2.0,
    sort_by: str = 'mean',
    threshold_percentile: float = 95.0,
    verbose: bool = True,
) -> Dict[str, Any]
```
Identyfikuje obrazy z anomalnymi odległościami przy użyciu wytrenowanych modeli.

#### `predict_lbp_anomalies_by_embedding_groups`
```python
.predict_lbp_anomalies_by_embedding_groups(
    patch_size: int = 5,
    visibility_threshold: float = 2.0,
    contamination: float = 0.1,
    random_state: int = 42,
    similarity_threshold: float = 0.95,
    similarity_k: int = DEFAULT_K_NEIGHBORS,
    n_estimators: int = 100,
    min_group_size: int = 10,
    device: Optional[str] = None,
    image_folder: Optional[str] = None,
    verbose: bool = True,
    precomputed_embeddings: Optional[np.ndarray] = None,
    precomputed_image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]
```
Isolation Forest per-grupa per-keypoint dla anomalii LBP.

#### `predict_lbp_anomalies_by_embedding_groups_zscore`
```python
.predict_lbp_anomalies_by_embedding_groups_zscore(
    patch_size: int = 5,
    visibility_threshold: float = 2.0,
    z_threshold: float = 2.0,
    similarity_threshold: float = 0.95,
    similarity_k: int = DEFAULT_K_NEIGHBORS,
    min_group_size: int = 10,
    ...
) -> Dict[str, Any]
```
Detekcja anomalii LBP metodą Z-score per-grupa per-keypoint.

#### `predict_lbp_anomalies_by_embedding_groups_ocsvm`
```python
.predict_lbp_anomalies_by_embedding_groups_ocsvm(
    patch_size: int = 5,
    visibility_threshold: float = 2.0,
    nu: float = 0.1,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    similarity_threshold: float = 0.95,
    similarity_k: int = DEFAULT_K_NEIGHBORS,
    min_group_size: int = 10,
    ...
) -> Dict[str, Any]
```
Detekcja anomalii LBP metodą One-Class SVM per-grupa per-keypoint.

#### `visualize_segmentation_and_keypoints`
```python
.visualize_segmentation_and_keypoints(
    output_folder: str,
    model_name: str = 'yolo26m-seg',
    target_class: Union[str, int] = 'person',
    score_threshold: float = 0.5,
    visibility_threshold: float = 2.0,
    image_paths: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[str]
```
Zapisuje wizualizacje łączące maski segmentacji z keypointami.

#### `draw_all_keypoints`
```python
.draw_all_keypoints(
    output_folder: str = 'output_keypoints',
    color: tuple = (0, 255, 0),
    keypoint_size: int = 4,
    visibility_threshold: float = 2.0,
    verbose: bool = True,
) -> List[str]
```
Rysuje keypointy na wszystkich obrazach datasetu i zapisuje wyniki.

#### `draw_all_keypoints_with_patches`
```python
.draw_all_keypoints_with_patches(
    output_folder: str = 'output_patches',
    patch_size: int = 32,
    keypoint_color: tuple = (0, 255, 0),
    patch_color: tuple = (255, 0, 0),
    visibility_threshold: float = 2.0,
    thickness: int = 2,
    verbose: bool = True,
) -> List[str]
```

### Metody statyczne

#### `find_near_duplicates` *(static)*
```python
YPSetValidation.find_near_duplicates(
    embeddings: np.ndarray,
    filenames: List[str],
    k: int = DEFAULT_K_NEIGHBORS,
    threshold: float = 0.99,
) -> List[Tuple[str, str, float]]
```
Wykrywa bliskie duplikaty na podstawie embeddingów CLIP.

#### `build_similarity_groups` *(static)*
```python
YPSetValidation.build_similarity_groups(
    embeddings: np.ndarray,
    threshold: float = 0.95,
    k: int = DEFAULT_K_NEIGHBORS,
) -> List[List[int]]
```
Buduje grupy podobnych obrazów przy użyciu FAISS.

#### `get_supported_segmentation_models` *(static)*
```python
YPSetValidation.get_supported_segmentation_models() -> List[str]
```
Zwraca listę obsługiwanych nazw modeli segmentacji.

---

## YPValidation_Test

`from utils.validation.weryfikacje import YPValidation_Test`  
`from utils import YPValidation_Test`

Kontener do gromadzenia i podsumowywania wyników eksperymentów walidacyjnych. Zawiera również metody klasowe do parsowania plików wynikowych.

### Konstruktor

```python
YPValidation_Test(
    name: str = "YPValidation_Test",
    metadata: Optional[Dict[str, Any]] = None,
)
```

### Właściwości

#### `results`
```python
.results -> List[Dict[str, Any]]
```
Płytka kopia listy zebranych wyników.

### Metody instancji

#### `add_result`
```python
.add_result(label: str, value: Any, **extra: Any) -> None
```
Dodaje pojedynczy wpis wynikowy.

#### `extend_results`
```python
.extend_results(entries: List[Dict[str, Any]]) -> None
```
Dodaje wiele wpisów naraz.

#### `clear`
```python
.clear() -> None
```
Usuwa wszystkie zebrane wyniki.

### Metody statyczne

#### `parse_results_txt_value` *(static)*
```python
YPValidation_Test.parse_results_txt_value(raw_value: str) -> Any
```
Konwertuje surowe pole pliku results.txt na obiekt Pythona.

#### `summarize_results_txt_entries` *(static)*
```python
YPValidation_Test.summarize_results_txt_entries(
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]
```
Agreguje statystyki z listy sparsowanych wpisów.

### Metody klasowe

#### `parse_results_txt_line` *(classmethod)*
```python
YPValidation_Test.parse_results_txt_line(cls, line: str) -> Dict[str, Any]
```
Parsuje pojedynczą linię pliku results.txt.

#### `parse_results_txt` *(classmethod)*
```python
YPValidation_Test.parse_results_txt(cls, results_txt_path: str) -> List[Dict[str, Any]]
```
Parsuje cały plik results.txt na listę rekordów.

#### `parse_detected_inconsistencies_txt` *(classmethod)*
```python
YPValidation_Test.parse_detected_inconsistencies_txt(
    cls, report_path: str,
) -> List[Dict[str, Any]]
```
Parsuje plain-text raport o niespójnościach na listę rekordów.

#### `parse_candidate_results_file` *(classmethod)*
```python
YPValidation_Test.parse_candidate_results_file(
    cls, candidate_path: str,
) -> List[Dict[str, Any]]
```
Parsuje plik kandydatów, auto-wykrywając obsługiwany format.

#### `summarize_results_txt` *(classmethod)*
```python
YPValidation_Test.summarize_results_txt(
    cls, results_txt_path: str,
) -> Dict[str, Any]
```
Parsuje i podsumowuje plik results.txt w jednym wywołaniu.

#### `compare_results_txt_entries` *(classmethod)*
```python
YPValidation_Test.compare_results_txt_entries(
    cls,
    reference_entries: List[Dict[str, Any]],
    candidate_entries: List[Dict[str, Any]],
    match_mode: str = 'auto',
) -> Dict[str, Any]
```
Porównuje dwie listy wpisów po znormalizowanej ścieżce obrazu i opcjonalnym `person_id`.

#### `compare_results_txt` *(classmethod)*
```python
YPValidation_Test.compare_results_txt(
    cls,
    reference_results_txt_path: str,
    candidate_results_txt_path: str,
) -> Dict[str, Any]
```
Parsuje i porównuje dwa pliki results.txt.

#### `parse_distance_anomaly_report` *(classmethod)*
```python
YPValidation_Test.parse_distance_anomaly_report(
    cls, distance_anomaly_report_path: str,
) -> List[Dict[str, Any]]
```
Parsuje zapisany raport rankingowy anomalii odległości.

#### `save_distance_anomaly_report` *(classmethod)*
```python
YPValidation_Test.save_distance_anomaly_report(
    cls,
    distance_anomaly_results: Dict[str, Any],
    output_path: str,
    only_anomalies: bool = False,
) -> str
```
Zapisuje wyniki rankingowe anomalii odległości do pliku tekstowego.

#### `compare_results_txt_to_distance_anomalies` *(classmethod)*
```python
YPValidation_Test.compare_results_txt_to_distance_anomalies(
    cls,
    reference_results_txt_path: str,
    distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    top_k: int,
    match_mode: str = 'auto',
    include_reference_fields: bool = False,
    reference_field_names: Optional[List[str]] = None,
) -> Dict[str, Any]
```
Porównuje plik results.txt z raportem rankingowym anomalii.

---

## YOLOPoseDataset & YOLOPoseImage

`from utils.datasets import YOLOPoseDataset, YOLOPoseImage`  
`from utils import YOLOPoseDataset, YOLOPoseImage`

---

### YOLOSample *(dataclass)*

Przykład zwracany przez `YOLOPoseDataset.__getitem__`. Obsługuje dostęp słownikowy dla wstecznej kompatybilności.

| Atrybut | Typ | Opis |
|---|---|---|
| `image_path` | `str` | Ścieżka do pliku obrazu |
| `label_path` | `str` | Ścieżka do pliku etykiet |
| `keypoints` | `np.ndarray` | Shape `(N_osób, K, 2 lub 3)` |

### YOLOPoseImage

Pojedynczy obraz w formacie YOLO Pose z adnotacjami keypointów.

#### Konstruktor

```python
YOLOPoseImage(
    image_path: Union[str, np.ndarray],
    label_path: str = None,
    include_visibility: bool = True,
    keypoint_format: str = 'auto',
)
```

#### Metody klasowe

```python
YOLOPoseImage.from_yaml_index(
    data_yaml: str,
    index: int,
    split: str = 'train',
    include_visibility: bool = True,
    keypoint_format: str = 'auto',
) -> 'YOLOPoseImage'
```
Ładuje obraz po indeksie z pliku data.yaml.

```python
YOLOPoseImage.from_txt_index(
    file_list_path: str,
    index: int,
    labels_dir: str = None,
    include_visibility: bool = True,
    keypoint_format: str = 'auto',
) -> 'YOLOPoseImage'
```
Ładuje obraz po indeksie z listy plików (train.txt).

#### Właściwości

| Właściwość | Zwraca | Opis |
|---|---|---|
| `num_keypoints` | `int` | Liczba keypointów |
| `image_size` | `tuple` | `(height, width)` |
| `has_visibility` | `bool` | Czy keypointy zawierają widoczność |

#### Metody instancji

```python
.get_sequential_distances(
    visibility_threshold: float = 2.0,
    distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray
```

```python
.get_sequential_angles(visibility_threshold: float = 2.0) -> np.ndarray
```

```python
.to_dict() -> Dict[str, Any]
```
Konwertuje do słownika kompatybilnego z YOLOPoseDataset.

---

### YOLOPoseDataset

Interfejs do pracy z datasetami YOLO Pose 1.0.

#### Konstruktor

```python
YOLOPoseDataset(
    dataset_file: str,
    labels_dir: str = None,
    keypoint_format: str = 'auto',
)
```

Akceptuje: `data.yaml`, `train.txt`, `val.txt`, `args.yaml`.

#### Właściwości

| Właściwość | Zwraca | Opis |
|---|---|---|
| `stats` | `dict` | Statystyki datasetu |

#### Metody instancji

```python
.__len__() -> int
.__getitem__(idx: int) -> YOLOSample
.get_keypoint_distribution(verbose: bool = True) -> dict
.get_keypoint_statistics(verbose: bool = True) -> dict
.get_sequential_distances(idx: int, visibility_threshold: float = 2.0) -> np.ndarray
.get_sequential_angles(idx: int, visibility_threshold: float = 2.0) -> np.ndarray
.get_all_keypoints() -> list
.get_all_images() -> list
.get_all_image_paths() -> list
.get_all_label_paths() -> list
.get_all_samples() -> list
```

**Zastępowanie keypointów (augmentacja):**

```python
.replace_angle_preserving_keypoints(
    sample_count: int,
    max_keypoints_to_shift: int,
    max_endpoint_shift: float,
    output_dataset_path: str = None,
    seed: int = None,
) -> list
```

```python
.replace_rigid_keypoints(
    sample_count: int,
    max_rotation_degrees: float,
    max_translation_distance: float,
    transform_mode: str = 'both',     # 'rotation' | 'translation' | 'both'
    output_dataset_path: str = None,
    seed: int = None,
) -> list
```

```python
.replace_axis_aligned_keypoints(
    sample_count: int,
    max_keypoints_to_shift: int,
    max_shift_distance: float,
    direction_mode: str = 'both',     # 'x' | 'y' | 'both'
    output_dataset_path: str = None,
    seed: int = None,
) -> list
```

**Eksport / zarządzanie:**

```python
.export_random_subset(
    sample_count: int,
    output_dataset_path: str,
    seed: int = None,
) -> str
```

```python
.export_dataset_partitions(
    part_count: int,
    output_root_dir: str,
    seed: int = None,
    shuffle: bool = True,
) -> List[str]
```

```python
.remove_duplicates(
    duplicates: Sequence[Sequence[Any]],
    output_dataset_path: Optional[str] = None,
    keep: str = 'first',
) -> Dict[str, Any]
```

```python
.remove_samples_by_image_paths(
    image_paths: Sequence[str],
    delete_files: bool = True,
) -> Dict[str, Any]
```

#### Metody statyczne

```python
YOLOPoseDataset.generate_train_txt_from_images(
    dataset_root: str,
    images_dir: str = 'images',
    output_file: str = None,
    recursive: bool = True,
) -> str
```
Generuje plik train.txt na podstawie obrazów w katalogu.

---

## Klasy wynikowe operacji

`from utils.datasets import AnglePreservingKeypointReplacement, RigidKeypointReplacement, AxisAlignedKeypointReplacement`

Dataclassy zwracane przez funkcje zastępowania keypointów. Każda ma:

| Właściwość | Zwraca | Opis |
|---|---|---|
| `changed_record` | `Dict[str, Union[str, int]]` | Identyfikator `{image_path, person_id}` zmienionej adnotacji |
| `console_output` | `str` | Opis tekstowy (to samo co drukowane w notebooku) |

`AnglePreservingKeypointReplacement` dodatkowo:

| Właściwość | Zwraca | Opis |
|---|---|---|
| `shift_magnitudes` | `np.ndarray` | Amplitudy przesunięć keypointów (norm. współrzędne) |

---

## Funkcje operacji na datasetach

`from utils.datasets import ...`  
`from utils import ...`

### `flatten_cvat_yolo_pose`
```python
flatten_cvat_yolo_pose(
    input_root: str,
    output_root: str,
    copy_data_yaml: bool = True,
) -> None
```
Spłaszcza zagnieżdżoną strukturę CVAT YOLO Pose do płaskiego datasetu.

### `merge_yolo_pose_datasets`
```python
merge_yolo_pose_datasets(
    dataset1_root: str,
    dataset2_root: str,
    output_root: str,
) -> None
```
Scala dwa datasety YOLO Pose. Kopiuje obrazy, etykiety i aktualizuje train.txt.

### `split_yolo_pose_dataset`
```python
split_yolo_pose_dataset(
    dataset_root: str,
    output_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    dataset_path: Optional[str] = None,
) -> None
```
Dzieli dataset na zbiory train/val w podanej proporcji.

### `convert_yolo_pose_to_cvat`
```python
convert_yolo_pose_to_cvat(
    dataset_root: str,
    output_zip_dir: str,
) -> None
```
Konwertuje dataset YOLO Pose do formatu CVAT (archiwa ZIP).

### `extract_image_subset`
```python
extract_image_subset(
    images_path: str,
    output_dir: str,
    a: int,
    b: int,
) -> None
```
Kopiuje obrazy o indeksach od `a` do `b` (włącznie) z folderu do `output_dir`.

### `generate_train_txt_from_images`
```python
generate_train_txt_from_images(
    dataset_root: str,
    images_dir: str = 'images',
    output_file: Optional[str] = None,
    recursive: bool = True,
) -> str
```
Generuje plik train.txt na podstawie plików obrazów w katalogu `images_dir`.

### `flatten_split_yolo_pose`
```python
flatten_split_yolo_pose(
    dataset_root: str,
    splits: Optional[List[str]] = None,
    images_dir: str = 'images',
    labels_dir: str = 'labels',
    remove_split_dirs: bool = True,
    output_root: Optional[str] = None,
) -> str
```
Spłaszcza podzielony dataset (`images/train/`, `images/val/`, ...) do płaskiej struktury.

### `replace_angle_preserving_keypoints`
```python
replace_angle_preserving_keypoints(
    dataset: Union[str, YOLOPoseDataset],
    sample_count: int,
    max_keypoints_to_shift: int,
    max_endpoint_shift: float,
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[AnglePreservingKeypointReplacement]
```
Zastępuje keypointy w datasecie zachowując kąty odcinków.

### `replace_rigid_keypoints`
```python
replace_rigid_keypoints(
    dataset: Union[str, YOLOPoseDataset],
    sample_count: int,
    max_rotation_degrees: float,
    max_translation_distance: float,
    transform_mode: str = 'both',
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[RigidKeypointReplacement]
```
Zastępuje keypointy rotacją, translacją lub obydwoma.

### `replace_axis_aligned_keypoints`
```python
replace_axis_aligned_keypoints(
    dataset: Union[str, YOLOPoseDataset],
    sample_count: int,
    max_keypoints_to_shift: int,
    max_shift_distance: float,
    direction_mode: str = 'both',
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[AxisAlignedKeypointReplacement]
```
Zastępuje keypointy przesuwając losowy podzbiór wzdłuż osi X/Y.

### `export_random_dataset_subset`
```python
export_random_dataset_subset(
    dataset: YOLOPoseDataset,
    sample_count: int,
    output_dataset_path: str,
    seed: Optional[int] = None,
) -> str
```
Eksportuje losowy podzbiór datasetu do nowego katalogu.

### `export_dataset_partitions`
```python
export_dataset_partitions(
    dataset: YOLOPoseDataset,
    part_count: int,
    output_root_dir: str,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> List[str]
```
Eksportuje dataset do `part_count` równych części.

### `remove_duplicate_samples_from_dataset`
```python
remove_duplicate_samples_from_dataset(
    dataset: YOLOPoseDataset,
    duplicates: Sequence[Sequence[Any]],
    output_dataset_path: Optional[str] = None,
    keep: str = 'first',
) -> Dict[str, Any]
```
Usuwa duplikaty opisane przez krotki z `find_near_duplicates`.

### `remove_samples_by_image_paths`
```python
remove_samples_by_image_paths(
    dataset: YOLOPoseDataset,
    image_paths: Iterable[str],
    delete_files: bool = True,
) -> Dict[str, Any]
```
Usuwa próbki datasetu identyfikowane przez ścieżki obrazów.

### `save_results_console_output`
```python
save_results_console_output(
    results: Iterable[object],
    output_path: str,
    include_details: bool = False,
    details_one_line: bool = False,
) -> str
```
Zapisuje wyniki zastępowania keypointów do pliku tekstowego (ten sam format co w konsoli).
