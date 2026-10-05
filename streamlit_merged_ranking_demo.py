"""Streamlit demo for merged ranking evaluation with mask-aware keypoint visualization."""

import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as import_error:
    cv2 = None
    CV2_IMPORT_ERROR = import_error
import numpy as np
import pandas as pd
import streamlit as st

from utils.validation.merged_ranking_evaluator import MergedRankingEvaluator
from utils.datasets.yolo_pose_dataset import YOLOPoseDataset
from utils.datasets import FORMAT_REGISTRY, KeypointDatasetAdapter


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".streamlit_cache"
OUTPUT_ROOT = APP_DIR / "streamlit_outputs"


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _download_file(url: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(urlparse(url).path).name or "downloaded_file"
    target = destination_dir / file_name
    urlretrieve(url, str(target))
    return target


def _find_dataset_file(search_root: Path) -> Path:
    candidates = [
        "data.yaml",
        "data.yml",
        "dataset.yaml",
        "dataset.yml",
        "train.txt",
    ]
    for candidate_name in candidates:
        matches = sorted(search_root.rglob(candidate_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Could not find data.yaml / dataset.yaml / train.txt in the provided dataset source."
    )


def _find_dataset_files(search_root: Path) -> list[Path]:
    candidates = [
        "data.yaml",
        "data.yml",
        "dataset.yaml",
        "dataset.yml",
        "train.txt",
    ]
    matches: list[Path] = []
    for candidate_name in candidates:
        matches.extend(sorted(search_root.rglob(candidate_name)))
    return matches


def _normalize_extracted_root(extract_dir: Path, max_depth: int = 4) -> Path:
    current = extract_dir
    for _ in range(max_depth):
        children = [child for child in current.iterdir() if child.is_dir()]
        files = [child for child in current.iterdir() if child.is_file()]
        has_dataset_markers = any(file.name.lower() in {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "train.txt"} for file in files)
        if has_dataset_markers:
            return current
        if len(children) == 1 and not files:
            current = children[0]
            continue
        return current
    return current


def _generate_train_txt_if_possible(dataset_root: Path) -> Optional[Path]:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not (images_dir.is_dir() and labels_dir.is_dir()):
        return None
    generated_train = YOLOPoseDataset.generate_train_txt_from_images(
        dataset_root=str(dataset_root),
        images_dir="images",
        recursive=True,
    )
    return Path(generated_train)


def _extract_uploaded_zip(uploaded_zip) -> Path:
    if uploaded_zip is None:
        raise ValueError("Please upload a dataset ZIP file.")

    if not str(uploaded_zip.name).lower().endswith(".zip"):
        raise ValueError("Uploaded file must be a .zip archive.")

    upload_dir = CACHE_DIR / "uploaded_archives"
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{Path(uploaded_zip.name).name}"
    zip_path = upload_dir / unique_name
    with open(zip_path, "wb") as zip_handle:
        uploaded_zip.seek(0)
        while True:
            chunk = uploaded_zip.read(8 * 1024 * 1024)
            if not chunk:
                break
            zip_handle.write(chunk)
        uploaded_zip.seek(0)

    extract_dir = CACHE_DIR / "extracted" / Path(unique_name).stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    return _normalize_extracted_root(extract_dir)


def _extract_zip_file(zip_path: Path) -> Path:
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("Provided file is not a .zip archive.")

    extract_dir = CACHE_DIR / "extracted" / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return _normalize_extracted_root(extract_dir)


def resolve_dataset_source(dataset_source: str) -> Path:
    value = dataset_source.strip()
    if not value:
        raise ValueError("Provide a dataset path or URL.")

    if _is_url(value):
        downloaded_path = _download_file(value, CACHE_DIR / "downloads")
        suffix = downloaded_path.suffix.lower()
        if suffix == ".zip":
            return _extract_zip_file(downloaded_path)
        if suffix in {".yaml", ".yml", ".txt", ".json"}:
            return downloaded_path
        raise ValueError("The URL must point to a .zip, .yaml/.yml, .txt, or .json file.")

    local_path = Path(value).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Path does not exist: {local_path}")

    if local_path.is_file() and local_path.suffix.lower() == ".zip":
        return _extract_zip_file(local_path)

    if local_path.is_dir():
        return local_path

    if local_path.suffix.lower() in {".yaml", ".yml", ".txt", ".json"}:
        return local_path

    raise ValueError("Local source must be a directory, .zip, .yaml/.yml/.txt/.json file, or a valid URL.")


def _resolve_yolo_dataset_file(dataset_source_path: Path) -> Path:
    if dataset_source_path.is_dir():
        normalized_root = _normalize_extracted_root(dataset_source_path)
        candidate_files = _find_dataset_files(normalized_root)

        generated_from_root = _generate_train_txt_if_possible(normalized_root)
        if generated_from_root is not None:
            candidate_files = [generated_from_root] + candidate_files

        # Prefer files that produce at least one valid image/label pair.
        for candidate in candidate_files:
            try:
                dataset = YOLOPoseDataset(str(candidate))
                if len(dataset) > 0:
                    return candidate
            except Exception:
                continue

        # Secondary fallback: check nested directories for images/ + labels/ and generate train.txt.
        for candidate_dir in sorted([p for p in normalized_root.rglob("*") if p.is_dir()]):
            generated_train = _generate_train_txt_if_possible(candidate_dir)
            if generated_train is None:
                continue
            try:
                dataset = YOLOPoseDataset(str(generated_train))
                if len(dataset) > 0:
                    return generated_train
            except Exception:
                continue

        # If descriptors exist but none were valid, return first one for a clear downstream error.
        if candidate_files:
            return candidate_files[0]

        raise FileNotFoundError(
            "Could not resolve a valid YOLO dataset from uploaded ZIP. "
            "Expected data.yaml/dataset.yaml/train.txt or images/ and labels/ directories."
        )

    if dataset_source_path.suffix.lower() in {".yaml", ".yml", ".txt"}:
        return dataset_source_path

    raise ValueError("YOLO Pose input must be a directory or a .yaml/.yml/.txt file.")


def _sample_key(image_path: str, person_id: int) -> Tuple[str, int]:
    return os.path.normcase(str(image_path)), int(person_id)


def _build_dataset_lookup(validator: Any) -> Dict[Tuple[str, int], Dict[str, np.ndarray]]:
    lookup: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for sample in validator.dataset:
        image = sample.get("image")
        image_path = sample.get("image_path")
        persons = validator._get_sample_person_keypoints(sample)
        for person_id, keypoints in enumerate(persons):
            lookup[_sample_key(image_path, person_id)] = {
                "image": image,
                "keypoints": np.asarray(keypoints, dtype=float),
            }
    return lookup


def _build_faulty_keypoint_lookup(evaluator: MergedRankingEvaluator) -> Dict[Tuple[str, int], Set[int]]:
    lookup: Dict[Tuple[str, int], Set[int]] = {}
    for key, meta in getattr(evaluator, "outside_lookup", {}).items():
        indices = set(int(idx) for idx in meta.get("outside_keypoint_indices", []))
        lookup[key] = indices
    return lookup


def _build_mask_lookup(evaluator: MergedRankingEvaluator) -> Dict[str, np.ndarray]:
    masks = getattr(evaluator, "segmentation_masks", None)
    if not isinstance(masks, dict):
        return {}

    image_paths = masks.get("image_paths") or []
    mask_list = masks.get("masks") or []
    output: Dict[str, np.ndarray] = {}
    for image_path, mask in zip(image_paths, mask_list):
        output[os.path.normcase(str(image_path))] = np.asarray(mask)
    return output


def annotate_image(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    faulty_indices: Set[int],
    mask: Optional[np.ndarray],
    visibility_threshold: float = 2.0,
) -> np.ndarray:
    result = image_bgr.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if mask is not None and np.asarray(mask).size > 0:
        mask_binary = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_binary.shape[:2] == result.shape[:2]:
            overlay = result.copy()
            overlay[mask_binary > 0] = (255, 128, 0)
            result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0.0)

    if keypoints.size == 0:
        return result

    height, width = result.shape[:2]
    points = np.asarray(keypoints, dtype=float)
    coords = points[:, :2].copy()
    if np.nanmax(coords) <= 1.0:
        coords[:, 0] *= width
        coords[:, 1] *= height

    for idx, (x_coord, y_coord) in enumerate(coords):
        if not np.isfinite(x_coord) or not np.isfinite(y_coord):
            continue
        if points.shape[1] >= 3 and float(points[idx, 2]) < visibility_threshold:
            continue

        px = int(np.clip(np.round(x_coord), 0, width - 1))
        py = int(np.clip(np.round(y_coord), 0, height - 1))
        is_faulty = idx in faulty_indices
        color = (0, 0, 255) if is_faulty else (0, 255, 0)
        radius = 7 if is_faulty else 5
        thickness = -1

        cv2.circle(result, (px, py), radius, color, thickness)
        label_color = (255, 255, 255) if is_faulty else (0, 0, 0)
        cv2.putText(
            result,
            str(idx),
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )

    return result


def run_ranking(
    dataset_source_path: Path,
    source_format: str,
    copy_images: bool,
    coco_annotations_dir: str,
    coco_images_dir: str,
    top_k: int,
    distance_weight: float,
    lbp_weight: float,
    segmentation_weight: float,
    segmentation_model_name: Optional[str],
    segmentation_target_class: str,
    segmentation_score_threshold: float,
    segmentation_mask_tolerance_px: int,
    distance_model_name: str,
    distance_sort_by: str,
    progress_callback=None,
) -> Tuple[Dict, MergedRankingEvaluator, Any, Optional[KeypointDatasetAdapter], Path]:
    # Import heavy validation stack lazily to keep Streamlit startup light.
    from utils.validation.set_validation import YPSetValidation

    def _emit(progress_value: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(int(progress_value), str(message))

    _emit(5, "Resolving dataset representation")
    adapter: Optional[KeypointDatasetAdapter] = None

    if source_format == "yolo_pose":
        dataset_file = _resolve_yolo_dataset_file(dataset_source_path)
        dataset = YOLOPoseDataset(str(dataset_file))
    else:
        input_root = str(dataset_source_path if dataset_source_path.is_dir() else dataset_source_path.parent)
        adapter_kwargs = {
            "source_format": source_format,
            "input_root": input_root,
            "copy_images": copy_images,
            "use_ultralytics_for_coco": True,
            "coco_annotations_dir": coco_annotations_dir.strip() or None,
            "coco_images_dir": coco_images_dir.strip() or None,
        }
        adapter = KeypointDatasetAdapter(**adapter_kwargs)
        dataset = adapter.as_yolo_dataset()
        dataset_file = Path(adapter.info().get("converted_dataset_file", ""))

    _emit(30, "Initializing dataset validator")
    validator = YPSetValidation(dataset)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_report = run_dir / "ranking_report.txt"

    _emit(45, "Preparing merged ranking evaluator")
    evaluator = MergedRankingEvaluator(
        validator=validator,
        report_paths={},
        top_k=top_k,
        output_path=str(output_report),
        distance_model_name=distance_model_name,
        distance_visibility_threshold=1.2,
        distance_threshold_percentile=95.0,
        distance_sort_by=distance_sort_by,
        segmentation_model_name=segmentation_model_name,
        segmentation_target_class=segmentation_target_class,
        segmentation_score_threshold=segmentation_score_threshold,
        segmentation_mask_tolerance_px=segmentation_mask_tolerance_px,
    )

    _emit(60, "Running ranking models")
    results = evaluator.run(distance_weight, lbp_weight, segmentation_weight)
    if not results:
        raise RuntimeError("Ranking was not generated. Check the selected weights.")

    _emit(85, "Saving ranking artifacts")
    ranking = results["combined_ranking"]
    ranking_df = pd.DataFrame(ranking)
    ranking_df.to_csv(run_dir / "combined_ranking.csv", index=False)

    with open(run_dir / "top_k_ranked_paths.txt", "w", encoding="utf-8") as handle:
        for row in ranking[:top_k]:
            handle.write(
                f"rank={row.get('rank')} | person_id={row.get('person_id')} | image={row.get('image_path')}\n"
            )

    results["run_dir"] = str(run_dir)
    _emit(100, "Merged ranking stage completed")
    return results, evaluator, validator, adapter, dataset_file


def save_annotated_outputs(
    run_dir: Path,
    ranking_rows,
    top_k: int,
    dataset_lookup: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    faulty_lookup: Dict[Tuple[str, int], Set[int]],
    mask_lookup: Dict[str, np.ndarray],
    progress_callback=None,
) -> pd.DataFrame:
    annotated_dir = run_dir / "annotated_top_k"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    written_rows = []
    selected_rows = ranking_rows[:top_k]
    total_rows = len(selected_rows)
    for index, row in enumerate(selected_rows, start=1):
        image_path = row.get("image_path")
        person_id = int(row.get("person_id", 0))
        key = _sample_key(image_path, person_id)
        payload = dataset_lookup.get(key)
        if payload is None:
            if progress_callback is not None:
                progress_callback(index, total_rows)
            continue

        image = payload["image"]
        keypoints = payload["keypoints"]
        mask = mask_lookup.get(os.path.normcase(str(image_path)))
        faulty = faulty_lookup.get(key, set())
        annotated = annotate_image(image, keypoints, faulty, mask)

        file_name = (
            f"rank_{int(row.get('rank', 0)):04d}"
            f"_person_{person_id:03d}"
            f"_{Path(str(image_path)).stem}.jpg"
        )
        output_path = annotated_dir / file_name
        cv2.imwrite(str(output_path), annotated)

        written_rows.append(
            {
                "rank": int(row.get("rank", 0)),
                "image_path": image_path,
                "person_id": person_id,
                "outside_keypoint_count": int(row.get("outside_keypoint_count", 0)),
                "annotated_file": str(output_path),
            }
        )

        if progress_callback is not None:
            progress_callback(index, total_rows)

    annotated_df = pd.DataFrame(written_rows)
    annotated_df.to_csv(run_dir / "annotated_manifest.csv", index=False)
    return annotated_df


def render_preview(annotated_df: pd.DataFrame, preview_limit: int) -> None:
    if annotated_df.empty:
        st.warning("No images available for preview.")
        return

    st.subheader("Annotated preview")
    for _, row in annotated_df.head(preview_limit).iterrows():
        image_path = row["annotated_file"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        caption = (
            f"rank={int(row['rank'])} | person_id={int(row['person_id'])} "
            f"| outside_keypoint_count={int(row['outside_keypoint_count'])}"
        )
        st.image(image_rgb, caption=caption, use_container_width=True)


st.set_page_config(page_title="A Unified Unsupervised Framework for Detecting Mislabeled Keypoints", layout="wide")

if cv2 is None:
    st.error(
        "OpenCV could not be imported. On Linux/Streamlit Cloud install system packages in packages.txt "
        "(for example: libgl1, libglib2.0-0)."
    )
    st.code(f"OpenCV import error: {CV2_IMPORT_ERROR}")
    st.stop()

st.title("A Unified Unsupervised Framework for Detecting Mislabeled Keypoints")

SEGMENTATION_CLASS_OPTIONS = {
    "yolo26n-seg": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "yolo26s-seg": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "yolo26m-seg": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "yolo26l-seg": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "yolo26x-seg": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "maskrcnn_resnet50_fpn": [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "deeplabv3_resnet50": [
        "person",
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "sheep",
    ],
    "lraspp_mobilenet_v3_large": [
        "person",
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "sheep",
    ],
    "fcn_resnet50": [
        "person",
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "sheep",
    ],
}

with st.sidebar:
    st.header("Input")

    st.caption("Dataset source")
    source_mode = st.radio(
        "Source mode",
        options=["Upload ZIP", "Path or URL"],
        index=0,
        help="Use ZIP upload for browser deployments or provide a direct server/local path/URL.",
    )

    dataset_source = ""
    uploaded_zip = st.file_uploader(
        "Upload dataset ZIP",
        type=["zip"],
        help=(
            "Upload a .zip archive containing your dataset. The archive should include at least one of: "
            "train.txt, data.yaml, or dataset.yaml (or files needed for selected input format conversion)."
        ),
        disabled=source_mode != "Upload ZIP",
    )

    if source_mode == "Path or URL":
        dataset_source = st.text_input(
            "Dataset path or URL",
            value="",
            help="Accepted: local directory, local .zip/.yaml/.yml/.txt/.json, or URL to those files.",
        )

    st.caption("ZIP upload supports archives with one extra top-level folder.")

    top_k = st.number_input("Number of samples to review", min_value=1, max_value=5000, value=450, step=1)
    st.caption("How many highest-ranked samples should be saved and previewed.")

    with st.expander("More settings", expanded=False):
        format_options = FORMAT_REGISTRY.list_formats()
        default_format_index = format_options.index("yolo_pose") if "yolo_pose" in format_options else 0
        source_format = st.selectbox(
            "Input dataset format",
            options=format_options,
            index=default_format_index,
            help="Select the format of your source dataset. Non-YOLO formats are converted internally.",
        )

        copy_images = st.checkbox(
            "Copy images during conversion",
            value=True,
            help="For non-YOLO formats: copy image files into the temporary YOLO-converted dataset.",
        )

        coco_annotations_dir = st.text_input(
            "COCO annotations directory (optional override)",
            value="",
            disabled=source_format != "coco_keypoints",
            help="Directory with COCO keypoints annotation JSON files. Leave empty to auto-resolve from dataset source.",
        )
        coco_images_dir = st.text_input(
            "COCO images directory (optional)",
            value="",
            disabled=source_format != "coco_keypoints",
            help="Optional path to source images to copy into converted YOLO dataset.",
        )

        st.subheader("Ranking weights")
        distance_weight = st.number_input("distance_weight", value=1.0, step=0.1, format="%.3f", help="Weight for distance-based anomaly score.")
        lbp_weight = st.number_input("lbp_weight", value=0.0, step=0.1, format="%.3f", help="Weight for LBP texture-based anomaly score. Set to 0 to skip CLIP embedding stage.")
        segmentation_weight = st.number_input("segmentation_weight", value=0.0, step=0.1, format="%.3f", help="Weight for segmentation-based score. Set to 0 to skip segmentation stage.")

        st.subheader("Distance")
        distance_model_name = st.selectbox(
            "distance_model_name",
            options=[
                "random_forest",
                "hist_gradient_boosting",
                "gradient_boosting",
                "linear",
                "ridge",
                "lasso",
                "elasticnet",
                "bayesian_ridge",
                "huber",
                "ransac",
                "svr",
                "decision_tree",
                "knn",
                "mlp",
            ],
            index=0,
            help="Regression model used to estimate keypoint-distance error.",
        )
        distance_sort_by = st.selectbox("distance_sort_by", options=["mean", "max"], index=0, help="How per-sample distance errors are aggregated.")

        st.subheader("Segmentation")
        use_segmentation = st.checkbox("Enable segmentation", value=True, help="If disabled, segmentation contribution is ignored.")
        segmentation_model_name = st.selectbox(
            "segmentation_model_name",
            options=[
                "yolo26n-seg",
                "yolo26s-seg",
                "yolo26m-seg",
                "yolo26l-seg",
                "yolo26x-seg",
                "maskrcnn_resnet50_fpn",
                "deeplabv3_resnet50",
                "lraspp_mobilenet_v3_large",
                "fcn_resnet50",
            ],
            index=2,
            disabled=not use_segmentation,
            help="Segmentation model used to build mask-based validation signals.",
        )
        class_options = SEGMENTATION_CLASS_OPTIONS.get(segmentation_model_name, ["person"])
        default_class_index = class_options.index("person") if "person" in class_options else 0
        segmentation_target_class = st.selectbox(
            "segmentation_target_class",
            options=class_options,
            index=default_class_index,
            disabled=not use_segmentation,
            help="Target class that defines object masks for keypoint-inside/outside checks.",
        )
        segmentation_score_threshold = st.number_input(
            "segmentation_score_threshold",
            value=0.5,
            step=0.05,
            format="%.2f",
            disabled=not use_segmentation,
            help="Minimum confidence score for accepted segmentation masks.",
        )
        segmentation_mask_tolerance_px = st.number_input(
            "segmentation_mask_tolerance_px",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
            disabled=not use_segmentation,
            help="Allowed pixel margin when testing whether a keypoint is outside a mask.",
        )

    run_button = st.button("Run ranking", type="primary")


if run_button:
    adapter: Optional[KeypointDatasetAdapter] = None
    try:
        if source_mode == "Upload ZIP":
            if uploaded_zip is None:
                st.error("Please upload a dataset ZIP file before running ranking.")
                st.stop()

        st.subheader("Pipeline progress")
        stage1_bar = st.progress(0, text="1/5 Dataset source: waiting")
        stage2_bar = st.progress(0, text="2/5 Merged ranking: waiting")
        stage3_bar = st.progress(0, text="3/5 Lookup tables: waiting")
        stage4_bar = st.progress(0, text="4/5 Annotated outputs: waiting")
        stage5_bar = st.progress(0, text="5/5 Render results: waiting")

        if source_mode == "Upload ZIP":
            stage1_bar.progress(20, text="1/5 Dataset source: extracting uploaded ZIP")
            dataset_source_path = _extract_uploaded_zip(uploaded_zip)
            stage1_bar.progress(100, text="1/5 Dataset source: ready")
        else:
            stage1_bar.progress(20, text="1/5 Dataset source: resolving path/URL")
            dataset_source_path = resolve_dataset_source(dataset_source)
            stage1_bar.progress(100, text="1/5 Dataset source: ready")

        stage2_bar.progress(5, text="2/5 Merged ranking: starting")

        resolved_seg_model = segmentation_model_name if use_segmentation else None
        resolved_seg_weight = segmentation_weight if use_segmentation else 0.0

        def _ranking_progress(current: int, message: str) -> None:
            stage2_bar.progress(max(1, min(100, int(current))), text=f"2/5 Merged ranking: {message}")

        results, evaluator, validator, adapter, dataset_file = run_ranking(
            dataset_source_path=dataset_source_path,
            source_format=source_format,
            copy_images=copy_images,
            coco_annotations_dir=coco_annotations_dir,
            coco_images_dir=coco_images_dir,
            top_k=int(top_k),
            distance_weight=float(distance_weight),
            lbp_weight=float(lbp_weight),
            segmentation_weight=float(resolved_seg_weight),
            segmentation_model_name=resolved_seg_model,
            segmentation_target_class=segmentation_target_class,
            segmentation_score_threshold=float(segmentation_score_threshold),
            segmentation_mask_tolerance_px=int(segmentation_mask_tolerance_px),
            distance_model_name=distance_model_name,
            distance_sort_by=distance_sort_by,
            progress_callback=_ranking_progress,
        )
        stage2_bar.progress(100, text="2/5 Merged ranking: done")

        ranking_rows = results.get("combined_ranking", [])
        run_dir = Path(results["run_dir"])

        stage3_bar.progress(20, text="3/5 Lookup tables: building")
        dataset_lookup = _build_dataset_lookup(validator)
        stage3_bar.progress(50, text="3/5 Lookup tables: faulty keypoints")
        faulty_lookup = _build_faulty_keypoint_lookup(evaluator)
        stage3_bar.progress(80, text="3/5 Lookup tables: segmentation masks")
        mask_lookup = _build_mask_lookup(evaluator)
        stage3_bar.progress(100, text="3/5 Lookup tables: done")

        stage4_bar.progress(5, text="4/5 Annotated outputs: starting")

        def _save_progress(current: int, total: int) -> None:
            if total <= 0:
                return
            local_ratio = current / total
            value = 5 + int(local_ratio * 95)
            stage4_bar.progress(min(value, 100), text=f"4/5 Annotated outputs: {current}/{total}")

        annotated_df = save_annotated_outputs(
            run_dir=run_dir,
            ranking_rows=ranking_rows,
            top_k=int(top_k),
            dataset_lookup=dataset_lookup,
            faulty_lookup=faulty_lookup,
            mask_lookup=mask_lookup,
            progress_callback=_save_progress,
        )
        stage4_bar.progress(100, text="4/5 Annotated outputs: done")

        stage5_bar.progress(30, text="5/5 Render results: preparing tables")

        st.success("Done.")
        st.write(f"Resolved source: {dataset_source_path}")
        st.write(f"Dataset file: {dataset_file}")
        st.write(f"Input format: {source_format}")
        st.write(f"Output dir: {run_dir}")

        ranking_df = pd.DataFrame(ranking_rows)
        st.subheader(f"Ranking (top {int(top_k)})")
        st.dataframe(ranking_df.head(int(top_k)), use_container_width=True)

        if not annotated_df.empty:
            st.subheader("Output files")
            st.dataframe(annotated_df, use_container_width=True)

        preview_default = min(10, int(top_k))
        preview_limit = st.slider(
            "Number of images to preview",
            min_value=1,
            max_value=max(1, min(50, int(top_k))),
            value=max(1, preview_default),
            step=1,
        )
        stage5_bar.progress(75, text="5/5 Render results: rendering previews")
        render_preview(annotated_df, preview_limit)
        stage5_bar.progress(100, text="5/5 Render results: done")

    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        if adapter is not None:
            adapter.close()
