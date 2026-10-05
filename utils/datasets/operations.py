"""Dataset manipulation operations (flatten, merge, split, convert)."""

from __future__ import annotations

import os
import shutil
import random
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..config import DEFAULT_VAL_RATIO, DEFAULT_RANDOM_SEED

if TYPE_CHECKING:
    from .yolo_pose_dataset import YOLOPoseDataset


@dataclass
class AnglePreservingKeypointReplacement:
    """In-place replacement result with preserved original segment angles."""

    image_path: str
    label_path: str
    person_id: int
    original_keypoints: np.ndarray
    replaced_keypoints: np.ndarray
    original_angles: np.ndarray
    replaced_angles: np.ndarray
    changed_keypoint_indices: List[int]
    shift_vectors: np.ndarray

    @property
    def shift_magnitudes(self) -> np.ndarray:
        """Return per-keypoint shift magnitudes in normalized coordinates."""
        shift_vectors = np.asarray(self.shift_vectors, dtype=float)
        if shift_vectors.size == 0:
            return np.zeros(0, dtype=float)
        return np.linalg.norm(shift_vectors, axis=1)

    @property
    def changed_record(self) -> Dict[str, Union[str, int]]:
        """Return the image/person identifier for the modified annotation."""
        return {
            'image_path': self.image_path,
            'person_id': self.person_id,
        }

    @property
    def console_output(self) -> str:
        """Return the same text representation typically printed in notebooks."""
        return str(self.changed_record)


@dataclass
class RigidKeypointReplacement:
    """In-place rigid transform result for a single person annotation."""

    image_path: str
    label_path: str
    person_id: int
    original_keypoints: np.ndarray
    replaced_keypoints: np.ndarray
    translation_vector: np.ndarray
    rotation_angle_degrees: float
    transform_mode: str

    @property
    def changed_record(self) -> Dict[str, Union[str, int]]:
        """Return the image/person identifier for the modified annotation."""
        return {
            'image_path': self.image_path,
            'person_id': self.person_id,
        }

    @property
    def console_output(self) -> str:
        """Return the same text representation typically printed in notebooks."""
        return f"{self.image_path} {self.person_id} {self.rotation_angle_degrees}"


@dataclass
class AxisAlignedKeypointReplacement:
    """In-place replacement result for axis-aligned per-keypoint shifts."""

    image_path: str
    label_path: str
    person_id: int
    original_keypoints: np.ndarray
    replaced_keypoints: np.ndarray
    changed_keypoint_indices: List[int]
    shift_vectors: np.ndarray
    direction_mode: str

    @property
    def changed_record(self) -> Dict[str, Union[str, int]]:
        """Return the image/person identifier for the modified annotation."""
        return {
            'image_path': self.image_path,
            'person_id': self.person_id,
        }

    @property
    def console_output(self) -> str:
        """Return the same text representation typically printed in notebooks."""
        return f"{self.image_path} {self.person_id} {self.changed_keypoint_indices}"


def _result_details_lines(result: object) -> List[str]:
    """Return human-readable detail lines for a replacement result object."""
    detail_lines: List[str] = []

    if hasattr(result, 'image_path'):
        detail_lines.append(f"image_path: {getattr(result, 'image_path')}")
    if hasattr(result, 'label_path'):
        detail_lines.append(f"label_path: {getattr(result, 'label_path')}")
    if hasattr(result, 'person_id'):
        detail_lines.append(f"person_id: {getattr(result, 'person_id')}")
    if hasattr(result, 'changed_keypoint_indices'):
        detail_lines.append(
            f"changed_keypoint_indices: {getattr(result, 'changed_keypoint_indices')}"
        )
    if hasattr(result, 'shift_vectors'):
        detail_lines.append(
            f"shift_vectors: {np.asarray(getattr(result, 'shift_vectors')).tolist()}"
        )
    if hasattr(result, 'shift_magnitudes'):
        detail_lines.append(
            f"shift_magnitudes: {np.asarray(getattr(result, 'shift_magnitudes')).tolist()}"
        )
    if hasattr(result, 'translation_vector'):
        detail_lines.append(
            f"translation_vector: {np.asarray(getattr(result, 'translation_vector')).tolist()}"
        )
    if hasattr(result, 'rotation_angle_degrees'):
        detail_lines.append(
            f"rotation_angle_degrees: {getattr(result, 'rotation_angle_degrees')}"
        )
    if hasattr(result, 'transform_mode'):
        detail_lines.append(f"transform_mode: {getattr(result, 'transform_mode')}")
    if hasattr(result, 'direction_mode'):
        detail_lines.append(f"direction_mode: {getattr(result, 'direction_mode')}")
    if hasattr(result, 'original_keypoints'):
        detail_lines.append(
            f"original_keypoints: {np.asarray(getattr(result, 'original_keypoints')).tolist()}"
        )
    if hasattr(result, 'replaced_keypoints'):
        detail_lines.append(
            f"replaced_keypoints: {np.asarray(getattr(result, 'replaced_keypoints')).tolist()}"
        )

    return detail_lines


def _result_details_one_line(result: object) -> str:
    """Return human-readable detail fields joined into a single line."""
    return " | ".join(_result_details_lines(result))


def _result_structured_one_line(result: object) -> str:
    """Return a stable one-line record for saving detailed replacement results."""
    return _result_details_one_line(result)


def save_results_console_output(
    results: Iterable[object],
    output_path: str,
    include_details: bool = False,
    details_one_line: bool = False,
) -> str:
    """Save replacement results to a text file using the same text shown in console prints.

    Args:
        results: Iterable of result objects returned by dataset replacement methods.
        output_path: Destination text file path.
        include_details: Whether to append detailed arrays and metadata for each result.
        details_one_line: When ``include_details`` is enabled, write each result and
            all details on a single line.

    Returns:
        Absolute path to the saved file.
    """
    output_path_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path_abs), exist_ok=True)

    lines = []
    for result_index, result in enumerate(results):
        line = getattr(result, 'console_output', str(result))
        if include_details:
            details = _result_structured_one_line(result)
            if details:
                if details_one_line:
                    lines.append(details)
                else:
                    lines.append(str(line))
                    lines.append(f"--- details {result_index} ---")
                    lines.extend(_result_details_lines(result))
            else:
                lines.append(str(line))
        else:
            lines.append(str(line))

    with open(output_path_abs, 'w', encoding='utf-8') as file_handle:
        file_handle.write("\n".join(lines) + ("\n" if lines else ""))

    return output_path_abs


def _compute_segment_angles(keypoints: np.ndarray) -> np.ndarray:
    """Return segment angles for all consecutive keypoint pairs."""
    from ..validation.image_validation import YPImageValidation

    if len(keypoints) < 2:
        return np.array([], dtype=[('angle_index', 'i4'), ('angle', 'f8')])

    return YPImageValidation.compute_sequential_angles(
        np.asarray(keypoints, dtype=float),
        visibility_threshold=float('-inf'),
    )


def _apply_angle_preserving_shift(
    keypoints: np.ndarray,
    keypoint_count_to_shift: int,
    max_endpoint_shift: float,
    min_shift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, List[int], np.ndarray]:
    """Shift one contiguous block of keypoints while preserving block geometry."""
    original = np.asarray(keypoints, dtype=float)
    replaced = original.copy()
    coords = replaced[:, :2]
    shift_vectors = np.zeros((len(replaced), 2), dtype=float)

    total_keypoints = len(coords)
    if keypoint_count_to_shift <= 0 or total_keypoints == 0:
        return replaced, [], shift_vectors

    shift_count = min(keypoint_count_to_shift, total_keypoints)
    block_options = []

    if shift_count == total_keypoints:
        if total_keypoints >= 2:
            prefix_direction = coords[1] - coords[0]
            suffix_direction = coords[-1] - coords[-2]
            if np.any(prefix_direction):
                block_options.append((slice(0, total_keypoints), prefix_direction, max_endpoint_shift))
            if np.any(suffix_direction):
                block_options.append((slice(0, total_keypoints), suffix_direction, max_endpoint_shift))
        else:
            block_options.append((slice(0, total_keypoints), np.array([1.0, 0.0], dtype=float), max_endpoint_shift))
    else:
        for block_start in range(0, total_keypoints - shift_count + 1):
            block_end = block_start + shift_count - 1
            block_slice = slice(block_start, block_end + 1)

            if block_start == 0:
                direction = coords[block_end + 1] - coords[block_end]
                max_shift = max_endpoint_shift if block_end == 0 else float(np.linalg.norm(direction))
                if np.any(direction) and max_shift > 0.0:
                    block_options.append((block_slice, direction, max_shift))
                continue

            if block_end == total_keypoints - 1:
                direction = coords[block_start] - coords[block_start - 1]
                max_shift = max_endpoint_shift if block_start == total_keypoints - 1 else float(np.linalg.norm(direction))
                if np.any(direction) and max_shift > 0.0:
                    block_options.append((block_slice, direction, max_shift))
                continue

            left_direction = coords[block_start] - coords[block_start - 1]
            right_direction = coords[block_end + 1] - coords[block_end]

            if np.any(left_direction):
                block_options.append((block_slice, left_direction, float(np.linalg.norm(left_direction))))
            if np.any(right_direction):
                block_options.append((block_slice, right_direction, float(np.linalg.norm(right_direction))))

    if not block_options:
        return replaced, [], shift_vectors

    block_slice, direction, max_shift = block_options[int(rng.integers(0, len(block_options)))]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0 or max_shift <= 0.0:
        return replaced, [], shift_vectors

    unit_direction = direction / direction_norm
    min_shift_for_option = min(float(min_shift), float(max_shift))
    shift_distance = float(rng.uniform(min_shift_for_option, max_shift))
    if rng.random() < 0.5:
        shift_distance *= -1.0

    applied_shift = unit_direction * shift_distance
    coords[block_slice] += applied_shift
    coords[:, :2] = np.clip(coords[:, :2], 0.0, 1.0)

    actual_shift_vectors = coords - original[:, :2]
    shift_vectors[:, :] = actual_shift_vectors

    changed_keypoint_indices = [
        index
        for index in range(len(original))
        if not np.allclose(original[index, :2], coords[index])
    ]

    replaced[:, :2] = coords
    return replaced, changed_keypoint_indices, shift_vectors


def _load_yolo_pose_label_entries(label_path: str) -> List[Dict[str, object]]:
    """Load raw YOLO pose label lines with metadata needed for rewriting."""
    from ..validation.image_validation import _parse_yolo_keypoint_values

    entries: List[Dict[str, object]] = []

    with open(label_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            parts = stripped_line.split()
            if len(parts) < 5:
                continue

            class_id = parts[0]
            bbox = [float(value) for value in parts[1:5]]
            keypoint_values = [float(value) for value in parts[5:]]

            entries.append({
                'class_id': class_id,
                'bbox': bbox,
                'keypoints': _parse_yolo_keypoint_values(
                    keypoint_values,
                    include_visibility=True,
                    keypoint_format='auto',
                ),
            })

    return entries


def _format_yolo_float(value: float) -> str:
    """Format numeric YOLO values compactly but deterministically."""
    return f"{float(value):.6f}"


def _write_yolo_pose_label_entries(label_path: str, entries: List[Dict[str, object]]) -> None:
    """Rewrite a YOLO pose label file from structured entries."""
    serialized_lines = []

    for entry in entries:
        class_id = str(entry['class_id'])
        bbox = [_format_yolo_float(value) for value in entry['bbox']]
        keypoints = np.asarray(entry['keypoints'], dtype=float)
        keypoint_values = [_format_yolo_float(value) for value in keypoints.reshape(-1)]
        serialized_lines.append(" ".join([class_id] + bbox + keypoint_values))

    with open(label_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write("\n".join(serialized_lines) + ("\n" if serialized_lines else ""))


def _resolve_source_data_yaml_path(dataset: 'YOLOPoseDataset') -> Optional[str]:
    """Return the most likely source data.yaml path for a dataset."""
    candidate_paths = [
        os.path.join(os.path.dirname(dataset.dataset_file), "data.yaml"),
        os.path.join(os.path.dirname(dataset.dataset_file), "data.yml"),
        os.path.join(dataset.dataset_dir, "data.yaml"),
        os.path.join(dataset.dataset_dir, "data.yml"),
        os.path.join(os.path.dirname(dataset.dataset_dir), "data.yaml"),
        os.path.join(os.path.dirname(dataset.dataset_dir), "data.yml"),
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _copy_data_yaml_for_export(dataset: 'YOLOPoseDataset', output_root: str) -> Optional[str]:
    """Copy data.yaml to exported dataset and rewrite split pointers when possible."""
    source_data_yaml = _resolve_source_data_yaml_path(dataset)
    if source_data_yaml is None:
        return None

    destination_data_yaml = os.path.join(output_root, os.path.basename(source_data_yaml))
    split_name = os.path.splitext(os.path.basename(dataset.dataset_file))[0]

    try:
        import yaml

        with open(source_data_yaml, 'r', encoding='utf-8') as file_handle:
            data_config = yaml.safe_load(file_handle) or {}

        if isinstance(data_config, dict):
            data_config['path'] = output_root
            for split_key in ('train', 'val', 'test'):
                if split_key == split_name or split_key in data_config:
                    data_config[split_key] = '.'

            with open(destination_data_yaml, 'w', encoding='utf-8') as file_handle:
                yaml.safe_dump(data_config, file_handle, sort_keys=False, allow_unicode=False)
            return destination_data_yaml
    except Exception:
        pass

    shutil.copy2(source_data_yaml, destination_data_yaml)
    return destination_data_yaml


def export_random_dataset_subset(
    dataset: 'YOLOPoseDataset',
    sample_count: int,
    output_dataset_path: str,
    seed: Optional[int] = None,
) -> str:
    """Export a random subset of dataset samples into a new dataset directory.

    The exported dataset preserves the original relative image layout, copies
    matching label files, writes a new split file, and copies ``data.yaml`` when
    it is available.

    Args:
        dataset: Source dataset instance.
        sample_count: Number of unique samples to export.
        output_dataset_path: Destination directory for the new dataset.
        seed: Optional RNG seed for reproducible sampling.

    Returns:
        Absolute path to the generated split file.

    Raises:
        ValueError: If ``sample_count`` is outside valid range.
        FileExistsError: If output path already exists and is not empty.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0")

    total_samples = len(dataset)
    if sample_count > total_samples:
        raise ValueError(
            f"sample_count ({sample_count}) cannot exceed dataset size ({total_samples})"
        )

    output_root = os.path.abspath(output_dataset_path)
    if os.path.exists(output_root) and os.listdir(output_root):
        raise FileExistsError(
            f"Output dataset path already exists and is not empty: {output_root}"
        )

    candidate_pairs = list(dataset._valid_pairs)
    random.Random(seed).shuffle(candidate_pairs)

    selected_records = []
    for image_path, label_path in candidate_pairs:
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        rel_path = os.path.relpath(image_path, dataset.dataset_dir)
        if rel_path is None:
            raise ValueError(f"Could not resolve relative path for image: {image_path}")
        selected_records.append((rel_path, image_path, label_path))

        if len(selected_records) == sample_count:
            break

    if len(selected_records) != sample_count:
        raise ValueError(
            "Not enough currently available image-label pairs to satisfy sample_count. "
            "Reload the dataset or lower sample_count."
        )

    os.makedirs(output_root, exist_ok=True)
    output_labels_dir = os.path.join(output_root, 'labels')
    os.makedirs(output_labels_dir, exist_ok=True)

    for image_rel_path, source_image_path, label_path in selected_records:
        destination_image_path = os.path.join(output_root, image_rel_path)
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        shutil.copy2(source_image_path, destination_image_path)

        destination_label_path = os.path.join(output_labels_dir, os.path.basename(label_path))
        shutil.copy2(label_path, destination_label_path)

    output_dataset_file = os.path.join(output_root, os.path.basename(dataset.dataset_file))
    with open(output_dataset_file, 'w', encoding='utf-8') as file_handle:
        file_handle.write(
            "\n".join(image_rel_path for image_rel_path, _, _ in selected_records)
            + ("\n" if selected_records else "")
        )

    _copy_data_yaml_for_export(dataset, output_root)
    return output_dataset_file


def export_dataset_partitions(
    dataset: 'YOLOPoseDataset',
    part_count: int,
    output_root_dir: str,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> List[str]:
    """Export the whole dataset into a chosen number of smaller parts.

    Each part is written into its own directory under ``output_root_dir`` as
    ``part_1``, ``part_2``, ..., preserving the original relative image layout,
    copying matching label files, writing a split file, and copying ``data.yaml``
    when available.

    Args:
        dataset: Source dataset instance.
        part_count: Number of parts to create.
        output_root_dir: Directory where part subdirectories will be created.
        seed: Optional RNG seed for reproducible shuffling.
        shuffle: Whether to shuffle samples before partitioning.

    Returns:
        List of absolute paths to generated split files, one per part.

    Raises:
        ValueError: If ``part_count`` is outside valid range.
        FileExistsError: If output path already exists and is not empty.
    """
    if part_count <= 0:
        raise ValueError("part_count must be greater than 0")

    candidate_pairs = list(dataset._valid_pairs)
    if shuffle:
        random.Random(seed).shuffle(candidate_pairs)

    selected_records = []
    for image_path, label_path in candidate_pairs:
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        rel_path = os.path.relpath(image_path, dataset.dataset_dir)
        if rel_path is None:
            raise ValueError(f"Could not resolve relative path for image: {image_path}")
        selected_records.append((rel_path, image_path, label_path))

    total_records = len(selected_records)
    if total_records == 0:
        raise ValueError("No currently available image-label pairs to export.")

    if part_count > total_records:
        raise ValueError(
            f"part_count ({part_count}) cannot exceed available sample count ({total_records})"
        )

    output_root = os.path.abspath(output_root_dir)
    if os.path.exists(output_root) and os.listdir(output_root):
        raise FileExistsError(
            f"Output root path already exists and is not empty: {output_root}"
        )

    os.makedirs(output_root, exist_ok=True)

    base_size = total_records // part_count
    remainder = total_records % part_count
    part_sizes = [base_size + (1 if index < remainder else 0) for index in range(part_count)]

    output_split_files = []
    start_index = 0
    for part_index, part_size in enumerate(part_sizes, start=1):
        end_index = start_index + part_size
        part_records = selected_records[start_index:end_index]
        start_index = end_index

        part_output_root = os.path.join(output_root, f"part_{part_index}")
        os.makedirs(part_output_root, exist_ok=True)
        output_labels_dir = os.path.join(part_output_root, 'labels')
        os.makedirs(output_labels_dir, exist_ok=True)

        for image_rel_path, source_image_path, label_path in part_records:
            destination_image_path = os.path.join(part_output_root, image_rel_path)
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            shutil.copy2(source_image_path, destination_image_path)

            destination_label_path = os.path.join(output_labels_dir, os.path.basename(label_path))
            shutil.copy2(label_path, destination_label_path)

        output_dataset_file = os.path.join(
            part_output_root,
            os.path.basename(dataset.dataset_file),
        )
        with open(output_dataset_file, 'w', encoding='utf-8') as file_handle:
            file_handle.write(
                "\n".join(image_rel_path for image_rel_path, _, _ in part_records)
                + ("\n" if part_records else "")
            )

        _copy_data_yaml_for_export(dataset, part_output_root)
        output_split_files.append(output_dataset_file)

    return output_split_files


def remove_duplicate_samples_from_dataset(
    dataset: 'YOLOPoseDataset',
    duplicates: Sequence[Sequence[Any]],
    output_dataset_path: Optional[str] = None,
    keep: str = 'first',
) -> Dict[str, Any]:
    """Remove duplicate samples identified from similarity tuples.

    The ``duplicates`` input is expected to be compatible with the output of
    ``YPSetValidation.find_near_duplicates``: an iterable of tuples like
    ``(file_a, file_b, similarity)``. Self-pairs are ignored. Symmetric pairs are
    merged into connected duplicate groups, and one sample per group is kept.

    Args:
        dataset: Source dataset instance.
        duplicates: Duplicate tuples or tuple-like sequences.
        output_dataset_path: When provided, write a cleaned dataset copy there.
            Otherwise, rewrite the current split file in place.
        keep: Which sample to keep per duplicate group: ``'first'`` or ``'last'``
            with respect to current dataset order.

    Returns:
        Summary dictionary describing kept, removed, skipped, and unresolved items.

    Raises:
        ValueError: If ``keep`` is invalid.
        FileExistsError: If the export path already exists and is not empty.
    """
    if keep not in {'first', 'last'}:
        raise ValueError("keep must be either 'first' or 'last'")

    dataset_records: List[Tuple[str, str, str]] = []
    for image_path, label_path in dataset._valid_pairs:
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        image_rel_path = os.path.relpath(image_path, dataset.dataset_dir)
        if image_rel_path is None:
            raise ValueError(f"Could not resolve relative path for image: {image_path}")
        dataset_records.append((image_rel_path, image_path, label_path))

    order_by_rel_path = {
        image_rel_path.replace('\\', '/'): index
        for index, (image_rel_path, _, _) in enumerate(dataset_records)
    }
    record_by_rel_path = {
        image_rel_path.replace('\\', '/'): (image_rel_path, image_path, label_path)
        for image_rel_path, image_path, label_path in dataset_records
    }
    record_by_abs_path = {
        os.path.normcase(os.path.normpath(image_path)).replace('\\', '/'): image_rel_path.replace('\\', '/')
        for image_rel_path, image_path, _ in dataset_records
    }

    basename_to_rel_paths: Dict[str, List[str]] = {}
    for image_rel_path, _, _ in dataset_records:
        normalized_rel_path = image_rel_path.replace('\\', '/')
        basename_to_rel_paths.setdefault(os.path.basename(normalized_rel_path), []).append(normalized_rel_path)

    def resolve_duplicate_identifier(identifier: Any) -> Optional[str]:
        if identifier is None:
            return None

        normalized_identifier = os.path.normcase(os.path.normpath(str(identifier))).replace('\\', '/')
        if normalized_identifier in record_by_abs_path:
            return record_by_abs_path[normalized_identifier]

        if normalized_identifier in record_by_rel_path:
            return normalized_identifier

        basename = os.path.basename(normalized_identifier)
        basename_matches = basename_to_rel_paths.get(basename, [])
        if len(basename_matches) == 1:
            return basename_matches[0]

        return None

    adjacency: Dict[str, Set[str]] = {}
    skipped_self_pairs: List[Dict[str, Any]] = []
    unresolved_pairs: List[Dict[str, Any]] = []

    for duplicate_entry in duplicates:
        if not isinstance(duplicate_entry, Sequence) or len(duplicate_entry) < 2:
            unresolved_pairs.append({'entry': duplicate_entry, 'reason': 'invalid_duplicate_entry'})
            continue

        file_a = duplicate_entry[0]
        file_b = duplicate_entry[1]
        similarity = duplicate_entry[2] if len(duplicate_entry) > 2 else None

        rel_path_a = resolve_duplicate_identifier(file_a)
        rel_path_b = resolve_duplicate_identifier(file_b)

        if rel_path_a is None or rel_path_b is None:
            unresolved_pairs.append(
                {
                    'entry': list(duplicate_entry),
                    'resolved_file_a': rel_path_a,
                    'resolved_file_b': rel_path_b,
                    'reason': 'unresolved_identifier',
                }
            )
            continue

        if rel_path_a == rel_path_b:
            skipped_self_pairs.append(
                {
                    'image_path': rel_path_a,
                    'similarity': similarity,
                }
            )
            continue

        adjacency.setdefault(rel_path_a, set()).add(rel_path_b)
        adjacency.setdefault(rel_path_b, set()).add(rel_path_a)

    visited: Set[str] = set()
    duplicate_groups: List[List[str]] = []
    for rel_path in adjacency:
        if rel_path in visited:
            continue

        stack = [rel_path]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited))

        component.sort(key=lambda path: order_by_rel_path[path])
        duplicate_groups.append(component)

    duplicate_groups.sort(key=lambda group: order_by_rel_path[group[0]])

    kept_rel_paths: Set[str] = set(record_by_rel_path.keys())
    removed_rel_paths: List[str] = []
    group_summaries: List[Dict[str, Any]] = []

    for group in duplicate_groups:
        keep_rel_path = group[0] if keep == 'first' else group[-1]
        current_removed = [path for path in group if path != keep_rel_path]
        for rel_path in current_removed:
            kept_rel_paths.discard(rel_path)
        removed_rel_paths.extend(current_removed)

        group_summaries.append(
            {
                'group_size': len(group),
                'keep_image_path': keep_rel_path,
                'removed_image_paths': current_removed,
                'group_image_paths': list(group),
            }
        )

    cleaned_records = [
        record_by_rel_path[image_rel_path.replace('\\', '/')]
        for image_rel_path, _, _ in dataset_records
        if image_rel_path.replace('\\', '/') in kept_rel_paths
    ]

    if output_dataset_path is None:
        with open(dataset.dataset_file, 'w', encoding='utf-8') as file_handle:
            file_handle.write(
                "\n".join(image_rel_path for image_rel_path, _, _ in cleaned_records)
                + ("\n" if cleaned_records else "")
            )

        dataset.image_rel_paths = [image_rel_path for image_rel_path, _, _ in cleaned_records]
        dataset._valid_pairs = dataset._validate_pairs()
        output_dataset_file = dataset.dataset_file
        output_mode = 'in_place_split_rewrite'
    else:
        output_root = os.path.abspath(output_dataset_path)
        if os.path.exists(output_root) and os.listdir(output_root):
            raise FileExistsError(
                f"Output dataset path already exists and is not empty: {output_root}"
            )

        os.makedirs(output_root, exist_ok=True)
        output_labels_dir = os.path.join(output_root, 'labels')
        os.makedirs(output_labels_dir, exist_ok=True)

        for image_rel_path, source_image_path, label_path in cleaned_records:
            destination_image_path = os.path.join(output_root, image_rel_path)
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            shutil.copy2(source_image_path, destination_image_path)

            destination_label_path = os.path.join(output_labels_dir, os.path.basename(label_path))
            shutil.copy2(label_path, destination_label_path)

        output_dataset_file = os.path.join(output_root, os.path.basename(dataset.dataset_file))
        with open(output_dataset_file, 'w', encoding='utf-8') as file_handle:
            file_handle.write(
                "\n".join(image_rel_path for image_rel_path, _, _ in cleaned_records)
                + ("\n" if cleaned_records else "")
            )

        _copy_data_yaml_for_export(dataset, output_root)
        output_mode = 'export_clean_copy'

    return {
        'output_mode': output_mode,
        'output_dataset_file': output_dataset_file,
        'duplicate_group_count': len(duplicate_groups),
        'removed_count': len(removed_rel_paths),
        'kept_count': len(cleaned_records),
        'removed_image_paths': removed_rel_paths,
        'kept_image_paths': [image_rel_path for image_rel_path, _, _ in cleaned_records],
        'duplicate_groups': group_summaries,
        'skipped_self_pairs': skipped_self_pairs,
        'unresolved_pairs': unresolved_pairs,
        'keep_strategy': keep,
    }


def remove_samples_by_image_paths(
    dataset: 'YOLOPoseDataset',
    image_paths: Iterable[str],
    delete_files: bool = True,
) -> Dict[str, Any]:
    """Remove dataset samples identified by image paths.

    Accepts absolute image paths or split-relative paths as stored in the
    dataset file. The split file is rewritten in place. When ``delete_files`` is
    true, matching image files and their corresponding label files are deleted
    from disk as well.

    Args:
        dataset: Dataset instance to modify.
        image_paths: Iterable of absolute or relative image paths to remove.
        delete_files: Whether to delete matching image and label files.

    Returns:
        Summary dictionary describing the performed removals.
    """
    from .yolo_pose_dataset import _candidate_label_paths

    requested_absolute_paths: Set[str] = set()
    requested_relative_paths: Set[str] = set()

    for image_path in image_paths:
        if image_path is None:
            continue
        normalized_input = str(image_path).strip()
        if not normalized_input:
            continue

        if os.path.isabs(normalized_input):
            requested_absolute_paths.add(
                os.path.normcase(os.path.abspath(normalized_input))
            )
        else:
            normalized_relative_path = normalized_input.replace('\\', '/')
            requested_relative_paths.add(normalized_relative_path)
            requested_absolute_paths.add(
                os.path.normcase(
                    os.path.abspath(os.path.join(dataset.dataset_dir, normalized_relative_path))
                )
            )

    removed_records: List[Dict[str, Optional[str]]] = []
    kept_rel_paths: List[str] = []

    for rel_path in dataset.image_rel_paths:
        normalized_rel_path = rel_path.replace('\\', '/')
        absolute_image_path = os.path.abspath(os.path.join(dataset.dataset_dir, rel_path))
        normalized_absolute_image_path = os.path.normcase(absolute_image_path)

        should_remove = (
            normalized_rel_path in requested_relative_paths
            or normalized_absolute_image_path in requested_absolute_paths
        )

        if not should_remove:
            kept_rel_paths.append(rel_path)
            continue

        label_path = None
        for candidate_label_path in _candidate_label_paths(rel_path, dataset.labels_dir):
            if os.path.exists(candidate_label_path):
                label_path = candidate_label_path
                break

        removed_records.append(
            {
                'image_rel_path': normalized_rel_path,
                'image_path': absolute_image_path,
                'label_path': label_path,
            }
        )

    matched_relative_paths = {
        record['image_rel_path']
        for record in removed_records
        if record['image_rel_path'] is not None
    }
    matched_absolute_paths = {
        os.path.normcase(record['image_path'])
        for record in removed_records
        if record['image_path'] is not None
    }

    unmatched_paths = [
        image_path
        for image_path in image_paths
        if image_path is not None and str(image_path).strip()
        and (
            (
                os.path.isabs(str(image_path).strip())
                and os.path.normcase(os.path.abspath(str(image_path).strip())) not in matched_absolute_paths
            )
            or (
                not os.path.isabs(str(image_path).strip())
                and str(image_path).strip().replace('\\', '/') not in matched_relative_paths
            )
        )
    ]

    with open(dataset.dataset_file, 'w', encoding='utf-8') as file_handle:
        file_handle.write("\n".join(kept_rel_paths) + ("\n" if kept_rel_paths else ""))

    deleted_images: List[str] = []
    deleted_labels: List[str] = []
    missing_images: List[str] = []
    missing_labels: List[str] = []

    if delete_files:
        for record in removed_records:
            image_path = record['image_path']
            label_path = record['label_path']

            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                deleted_images.append(image_path)
            elif image_path:
                missing_images.append(image_path)

            if label_path and os.path.exists(label_path):
                os.remove(label_path)
                deleted_labels.append(label_path)
            elif label_path:
                missing_labels.append(label_path)

    dataset.image_rel_paths = kept_rel_paths
    dataset._valid_pairs = dataset._validate_pairs()

    return {
        'dataset_file': dataset.dataset_file,
        'requested_count': len(requested_absolute_paths | requested_relative_paths),
        'matched_count': len(removed_records),
        'removed_count': len(removed_records),
        'kept_count': len(kept_rel_paths),
        'delete_files': delete_files,
        'deleted_images_count': len(deleted_images),
        'deleted_labels_count': len(deleted_labels),
        'removed_image_paths': [record['image_path'] for record in removed_records],
        'removed_label_paths': [record['label_path'] for record in removed_records if record['label_path']],
        'unmatched_paths': unmatched_paths,
        'missing_images': missing_images,
        'missing_labels': missing_labels,
    }


def _persist_replacement_dataset(
    dataset: 'YOLOPoseDataset',
    label_entries_cache: Dict[str, List[Dict[str, object]]],
    output_dataset_path: Optional[str],
) -> None:
    """Write modified labels either in place or into a copied dataset."""
    if output_dataset_path is None:
        for label_path, entries in label_entries_cache.items():
            _write_yolo_pose_label_entries(label_path, entries)
        return

    output_root = os.path.abspath(output_dataset_path)
    output_labels_dir = os.path.join(output_root, 'labels')
    os.makedirs(output_labels_dir, exist_ok=True)

    for image_rel_path in dataset.image_rel_paths:
        source_image_path = os.path.join(dataset.dataset_dir, image_rel_path)
        if not os.path.exists(source_image_path):
            continue
        destination_image_path = os.path.join(output_root, image_rel_path)
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        shutil.copy2(source_image_path, destination_image_path)

    output_dataset_file = os.path.join(output_root, os.path.basename(dataset.dataset_file))
    os.makedirs(os.path.dirname(output_dataset_file), exist_ok=True)
    with open(output_dataset_file, 'w', encoding='utf-8') as file_handle:
        file_handle.write("\n".join(dataset.image_rel_paths) + ("\n" if dataset.image_rel_paths else ""))

    source_label_paths = dataset.get_all_label_paths()
    for source_label_path in source_label_paths:
        entries = label_entries_cache.get(source_label_path)
        if entries is None:
            entries = _load_yolo_pose_label_entries(source_label_path)
        destination_label_path = os.path.join(output_labels_dir, os.path.basename(source_label_path))
        _write_yolo_pose_label_entries(destination_label_path, entries)

    _copy_data_yaml_for_export(dataset, output_root)


def _build_person_records(dataset: 'YOLOPoseDataset') -> List[Dict[str, object]]:
    """Collect per-person records from a YOLOPoseDataset."""
    all_persons: List[Dict[str, object]] = []
    for sample in dataset:
        image_rel_path = os.path.relpath(sample['image_path'], dataset.dataset_dir)
        label_name = os.path.basename(sample['label_path'])
        for person_id, person_keypoints in enumerate(sample['all_keypoints']):
            person_keypoints_array = np.asarray(person_keypoints, dtype=float)
            all_persons.append({
                'image_path': sample['image_path'],
                'image_rel_path': image_rel_path,
                'label_path': sample['label_path'],
                'label_name': label_name,
                'person_id': person_id,
                'keypoints': person_keypoints_array,
            })
    return all_persons


def _compute_yolo_bbox_from_keypoints(keypoints: np.ndarray) -> List[float]:
    """Compute YOLO bbox [cx, cy, w, h] from normalized keypoints."""
    keypoints_array = np.asarray(keypoints, dtype=float)
    if keypoints_array.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    coords = np.clip(keypoints_array[:, :2], 0.0, 1.0)
    x_min = float(np.min(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    x_max = float(np.max(coords[:, 0]))
    y_max = float(np.max(coords[:, 1]))

    return [
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        max(0.0, x_max - x_min),
        max(0.0, y_max - y_min),
    ]


def _build_translation_vector(max_translation_distance: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a random 2D translation vector with bounded magnitude."""
    if max_translation_distance <= 0.0:
        return np.zeros(2, dtype=float)

    angle_radians = float(rng.uniform(-np.pi, np.pi))
    distance = float(rng.uniform(0.0, max_translation_distance))
    return np.array([
        np.cos(angle_radians) * distance,
        np.sin(angle_radians) * distance,
    ], dtype=float)


def _rotate_coords(coords: np.ndarray, angle_degrees: float, center: np.ndarray) -> np.ndarray:
    """Rotate 2D coordinates around the provided center."""
    angle_radians = np.deg2rad(float(angle_degrees))
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)],
    ], dtype=float)
    return (coords - center) @ rotation_matrix.T + center


def _apply_rigid_transform(
    keypoints: np.ndarray,
    max_rotation_degrees: float,
    max_translation_distance: float,
    transform_mode: str,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply translation, rotation, or both while keeping keypoints inside frame."""
    mode = str(transform_mode).lower()
    allowed_modes = {"translate", "rotate", "both"}
    if mode not in allowed_modes:
        raise ValueError(f"transform_mode must be one of {sorted(allowed_modes)}")

    original = np.asarray(keypoints, dtype=float)
    coords = original[:, :2]
    center = coords.mean(axis=0)

    translation_scale = 1.0
    rotation_scale = 1.0

    for _ in range(max_attempts):
        translation_vector = np.zeros(2, dtype=float)
        rotation_angle_degrees = 0.0

        if mode in {"translate", "both"} and max_translation_distance > 0.0:
            translation_vector = _build_translation_vector(
                max_translation_distance * translation_scale,
                rng,
            )
        if mode in {"rotate", "both"} and max_rotation_degrees > 0.0:
            rotation_angle_degrees = float(
                rng.uniform(-max_rotation_degrees * rotation_scale, max_rotation_degrees * rotation_scale)
            )

        transformed_coords = coords.copy()
        if mode in {"rotate", "both"} and rotation_angle_degrees != 0.0:
            transformed_coords = _rotate_coords(transformed_coords, rotation_angle_degrees, center=center)
        if mode in {"translate", "both"} and np.any(translation_vector):
            transformed_coords = transformed_coords + translation_vector

        if np.all((transformed_coords >= 0.0) & (transformed_coords <= 1.0)):
            replaced = original.copy()
            replaced[:, :2] = transformed_coords
            return replaced, translation_vector, rotation_angle_degrees

        translation_scale *= 0.8
        rotation_scale *= 0.8

    replaced = original.copy()
    return replaced, np.zeros(2, dtype=float), 0.0


def _sample_axis_aligned_shift(
    max_shift_distance: float,
    direction_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an axis-aligned shift vector for one keypoint."""
    mode = str(direction_mode).lower()
    allowed_modes = {"horizontal", "vertical", "both"}
    if mode not in allowed_modes:
        raise ValueError(f"direction_mode must be one of {sorted(allowed_modes)}")

    if max_shift_distance <= 0.0:
        return np.zeros(2, dtype=float)

    if mode == "horizontal":
        return np.array([rng.uniform(-max_shift_distance, max_shift_distance), 0.0], dtype=float)
    if mode == "vertical":
        return np.array([0.0, rng.uniform(-max_shift_distance, max_shift_distance)], dtype=float)

    axis_x = float(rng.uniform(-max_shift_distance, max_shift_distance))
    axis_y = float(rng.uniform(-max_shift_distance, max_shift_distance))
    return np.array([axis_x, axis_y], dtype=float)


def _apply_axis_aligned_keypoint_shift(
    keypoints: np.ndarray,
    max_keypoints_to_shift: int,
    max_shift_distance: float,
    direction_mode: str,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> tuple[np.ndarray, List[int], np.ndarray]:
    """Shift a random subset of keypoints along axes while staying in frame."""
    original = np.asarray(keypoints, dtype=float)
    replaced = original.copy()

    max_indices_to_shift = min(max_keypoints_to_shift, len(original))
    selected_count = int(rng.integers(1, max_indices_to_shift + 1))
    selected_indices = sorted(
        int(index)
        for index in rng.choice(len(original), size=selected_count, replace=False)
    )

    shift_vectors = np.zeros((len(original), 2), dtype=float)

    for keypoint_index in selected_indices:
        coords = replaced[keypoint_index, :2].copy()
        shift_scale = 1.0
        applied_shift = np.zeros(2, dtype=float)

        for _ in range(max_attempts):
            shift_vector = _sample_axis_aligned_shift(
                max_shift_distance=max_shift_distance * shift_scale,
                direction_mode=direction_mode,
                rng=rng,
            )
            shifted_coords = coords + shift_vector
            if np.all((shifted_coords >= 0.0) & (shifted_coords <= 1.0)):
                replaced[keypoint_index, :2] = shifted_coords
                applied_shift = shift_vector
                break
            shift_scale *= 0.8

        shift_vectors[keypoint_index] = applied_shift

    changed_keypoint_indices = [
        index
        for index in range(len(original))
        if not np.allclose(original[index, :2], replaced[index, :2])
    ]
    return replaced, changed_keypoint_indices, shift_vectors


def replace_angle_preserving_keypoints(
    dataset: Union[str, 'YOLOPoseDataset'],
    sample_count: int,
    max_keypoints_to_shift: int,
    max_endpoint_shift: float,
    min_shift: float = 0.0,
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[AnglePreservingKeypointReplacement]:
    """Replace existing dataset keypoints in place while preserving segment angles.

    The function loads all annotated persons from the dataset, selects existing
    persons without replacement, and rewrites their label lines in place. Exactly
    ``max_keypoints_to_shift`` keypoints are moved in each replaced annotation,
    capped by the number of available keypoints. The move is applied to one
    contiguous block, which preserves the relative geometry inside that shifted
    block. The dataset size does not increase.

    Args:
        dataset: Either a ``YOLOPoseDataset`` instance or a path to ``data.yaml``
            or ``train.txt``.
        sample_count: Number of existing person annotations to replace.
        max_keypoints_to_shift: Exact number of keypoints to move in a single
            replaced annotation, capped by the number of available keypoints.
        max_endpoint_shift: Maximum shift magnitude for edge keypoints.
        min_shift: Minimum shift magnitude to apply. If this value is larger than
            the available maximum for a selected block, the method uses that
            available maximum.
        labels_dir: Optional explicit labels directory.
        output_dataset_path: Optional path where a copied dataset with modified
            labels should be written. When omitted, labels are modified in place.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of replacement results. Each result contains original keypoints,
        replaced keypoints, original angles, replaced angles, and shifted indices.

    Raises:
        ValueError: If the dataset has no valid persons or input parameters are invalid.
    """
    from .yolo_pose_dataset import YOLOPoseDataset

    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    if max_keypoints_to_shift <= 0:
        raise ValueError("max_keypoints_to_shift must be > 0")
    if max_endpoint_shift < 0:
        raise ValueError("max_endpoint_shift must be >= 0")
    if min_shift < 0:
        raise ValueError("min_shift must be >= 0")

    if isinstance(dataset, YOLOPoseDataset):
        yolo_dataset = dataset
    else:
        yolo_dataset = YOLOPoseDataset(dataset, labels_dir=labels_dir)

    rng = np.random.default_rng(seed)

    all_persons = [
        person_record
        for person_record in _build_person_records(yolo_dataset)
        if person_record['keypoints'].ndim == 2 and len(person_record['keypoints']) >= 2
    ]

    if not all_persons:
        raise ValueError("Dataset does not contain any person with at least 2 keypoints")

    if sample_count > len(all_persons):
        raise ValueError(
            f"sample_count={sample_count} exceeds available persons={len(all_persons)}"
        )

    selected_person_indices = rng.choice(len(all_persons), size=sample_count, replace=False)
    label_entries_cache: Dict[str, List[Dict[str, object]]] = {}
    replacements: List[AnglePreservingKeypointReplacement] = []

    for selected_person_index in selected_person_indices:
        source_person = all_persons[int(selected_person_index)]
        original_keypoints = source_person['keypoints'].copy()
        exact_shift_count = min(max_keypoints_to_shift, len(original_keypoints))

        replaced_keypoints, changed_keypoint_indices, shift_vectors = _apply_angle_preserving_shift(
            original_keypoints,
            keypoint_count_to_shift=exact_shift_count,
            max_endpoint_shift=max_endpoint_shift,
            min_shift=min_shift,
            rng=rng,
        )

        label_path = source_person['label_path']
        if label_path not in label_entries_cache:
            label_entries_cache[label_path] = _load_yolo_pose_label_entries(label_path)

        label_entries_cache[label_path][source_person['person_id']]['keypoints'] = replaced_keypoints.copy()
        label_entries_cache[label_path][source_person['person_id']]['bbox'] = _compute_yolo_bbox_from_keypoints(
            replaced_keypoints
        )

        result_image_path = source_person['image_path']
        result_label_path = label_path
        if output_dataset_path is not None:
            output_root = os.path.abspath(output_dataset_path)
            result_image_path = os.path.join(output_root, source_person['image_rel_path'])
            result_label_path = os.path.join(output_root, 'labels', source_person['label_name'])

        replacements.append(
            AnglePreservingKeypointReplacement(
                image_path=result_image_path,
                label_path=result_label_path,
                person_id=source_person['person_id'],
                original_keypoints=original_keypoints,
                replaced_keypoints=replaced_keypoints,
                original_angles=_compute_segment_angles(original_keypoints),
                replaced_angles=_compute_segment_angles(replaced_keypoints),
                changed_keypoint_indices=changed_keypoint_indices,
                shift_vectors=shift_vectors,
            )
        )

    _persist_replacement_dataset(
        yolo_dataset,
        label_entries_cache=label_entries_cache,
        output_dataset_path=output_dataset_path,
    )

    if isinstance(dataset, YOLOPoseDataset) and output_dataset_path is None:
        dataset._valid_pairs = dataset._validate_pairs()

    return replacements


def replace_rigid_keypoints(
    dataset: Union[str, 'YOLOPoseDataset'],
    sample_count: int,
    max_rotation_degrees: float,
    max_translation_distance: float,
    transform_mode: str = 'both',
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[RigidKeypointReplacement]:
    """Replace existing person annotations using a rigid transform in place.

    Applies translation, rotation, or both to all keypoints of selected persons.
    The relative geometry of the skeleton is preserved because the transform is
    rigid for the whole point set. Modified label files are rewritten in place and
    the dataset size does not change.

    Args:
        dataset: Either a ``YOLOPoseDataset`` instance or a path to ``data.yaml``
            or ``train.txt``.
        sample_count: Number of existing person annotations to replace.
        max_rotation_degrees: Maximum absolute rotation angle in degrees.
        max_translation_distance: Maximum translation magnitude in normalized units.
        transform_mode: One of ``'translate'``, ``'rotate'`` or ``'both'``.
        labels_dir: Optional explicit labels directory.
        output_dataset_path: Optional path where a copied dataset with modified
            labels should be written. When omitted, labels are modified in place.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of rigid replacement results with ``image_path`` and ``person_id``.
    """
    from .yolo_pose_dataset import YOLOPoseDataset

    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    if max_rotation_degrees < 0:
        raise ValueError("max_rotation_degrees must be >= 0")
    if max_translation_distance < 0:
        raise ValueError("max_translation_distance must be >= 0")
    if max_rotation_degrees == 0 and max_translation_distance == 0:
        raise ValueError("At least one of max_rotation_degrees or max_translation_distance must be > 0")

    if isinstance(dataset, YOLOPoseDataset):
        yolo_dataset = dataset
    else:
        yolo_dataset = YOLOPoseDataset(dataset, labels_dir=labels_dir)

    rng = np.random.default_rng(seed)

    all_persons = [
        person_record
        for person_record in _build_person_records(yolo_dataset)
        if person_record['keypoints'].ndim == 2 and len(person_record['keypoints']) >= 2
    ]

    if not all_persons:
        raise ValueError("Dataset does not contain any person with at least 2 keypoints")

    if sample_count > len(all_persons):
        raise ValueError(
            f"sample_count={sample_count} exceeds available persons={len(all_persons)}"
        )

    selected_person_indices = rng.choice(len(all_persons), size=sample_count, replace=False)
    label_entries_cache: Dict[str, List[Dict[str, object]]] = {}
    replacements: List[RigidKeypointReplacement] = []

    for selected_person_index in selected_person_indices:
        source_person = all_persons[int(selected_person_index)]
        original_keypoints = source_person['keypoints'].copy()
        replaced_keypoints, translation_vector, rotation_angle_degrees = _apply_rigid_transform(
            original_keypoints,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_distance=max_translation_distance,
            transform_mode=transform_mode,
            rng=rng,
        )

        label_path = source_person['label_path']
        if label_path not in label_entries_cache:
            label_entries_cache[label_path] = _load_yolo_pose_label_entries(label_path)

        label_entries_cache[label_path][source_person['person_id']]['keypoints'] = replaced_keypoints.copy()
        label_entries_cache[label_path][source_person['person_id']]['bbox'] = _compute_yolo_bbox_from_keypoints(
            replaced_keypoints
        )

        result_image_path = source_person['image_path']
        result_label_path = label_path
        if output_dataset_path is not None:
            output_root = os.path.abspath(output_dataset_path)
            result_image_path = os.path.join(output_root, source_person['image_rel_path'])
            result_label_path = os.path.join(output_root, 'labels', source_person['label_name'])

        replacements.append(
            RigidKeypointReplacement(
                image_path=result_image_path,
                label_path=result_label_path,
                person_id=source_person['person_id'],
                original_keypoints=original_keypoints,
                replaced_keypoints=replaced_keypoints,
                translation_vector=translation_vector,
                rotation_angle_degrees=rotation_angle_degrees,
                transform_mode=str(transform_mode).lower(),
            )
        )

    _persist_replacement_dataset(
        yolo_dataset,
        label_entries_cache=label_entries_cache,
        output_dataset_path=output_dataset_path,
    )

    if isinstance(dataset, YOLOPoseDataset) and output_dataset_path is None:
        dataset._valid_pairs = dataset._validate_pairs()

    return replacements


def replace_axis_aligned_keypoints(
    dataset: Union[str, 'YOLOPoseDataset'],
    sample_count: int,
    max_keypoints_to_shift: int,
    max_shift_distance: float,
    direction_mode: str = 'both',
    labels_dir: Optional[str] = None,
    output_dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[AxisAlignedKeypointReplacement]:
    """Replace existing person annotations by shifting random keypoints along axes.

    Selects existing persons without replacement and modifies only a random subset
    of their keypoints. Each modified keypoint is shifted left/right, up/down, or
    in both axes depending on ``direction_mode``. Label files are rewritten in
    place and the dataset size does not change.

    Args:
        dataset: Either a ``YOLOPoseDataset`` instance or a path to ``data.yaml``
            or ``train.txt``.
        sample_count: Number of existing person annotations to replace.
        max_keypoints_to_shift: Maximum number of keypoints to shift per person.
        max_shift_distance: Maximum per-axis shift magnitude in normalized units.
        direction_mode: One of ``'horizontal'``, ``'vertical'`` or ``'both'``.
        labels_dir: Optional explicit labels directory.
        output_dataset_path: Optional path where a copied dataset with modified
            labels should be written. When omitted, labels are modified in place.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of axis-aligned replacement results with ``image_path`` and ``person_id``.
    """
    from .yolo_pose_dataset import YOLOPoseDataset

    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    if max_keypoints_to_shift <= 0:
        raise ValueError("max_keypoints_to_shift must be > 0")
    if max_shift_distance < 0:
        raise ValueError("max_shift_distance must be >= 0")
    if max_shift_distance == 0:
        raise ValueError("max_shift_distance must be > 0")

    if isinstance(dataset, YOLOPoseDataset):
        yolo_dataset = dataset
    else:
        yolo_dataset = YOLOPoseDataset(dataset, labels_dir=labels_dir)

    rng = np.random.default_rng(seed)

    all_persons = [
        person_record
        for person_record in _build_person_records(yolo_dataset)
        if person_record['keypoints'].ndim == 2 and len(person_record['keypoints']) >= 1
    ]

    if not all_persons:
        raise ValueError("Dataset does not contain any person annotations")

    if sample_count > len(all_persons):
        raise ValueError(
            f"sample_count={sample_count} exceeds available persons={len(all_persons)}"
        )

    selected_person_indices = rng.choice(len(all_persons), size=sample_count, replace=False)
    label_entries_cache: Dict[str, List[Dict[str, object]]] = {}
    replacements: List[AxisAlignedKeypointReplacement] = []

    for selected_person_index in selected_person_indices:
        source_person = all_persons[int(selected_person_index)]
        original_keypoints = source_person['keypoints'].copy()
        replaced_keypoints, changed_keypoint_indices, shift_vectors = _apply_axis_aligned_keypoint_shift(
            original_keypoints,
            max_keypoints_to_shift=max_keypoints_to_shift,
            max_shift_distance=max_shift_distance,
            direction_mode=direction_mode,
            rng=rng,
        )

        label_path = source_person['label_path']
        if label_path not in label_entries_cache:
            label_entries_cache[label_path] = _load_yolo_pose_label_entries(label_path)

        label_entries_cache[label_path][source_person['person_id']]['keypoints'] = replaced_keypoints.copy()
        label_entries_cache[label_path][source_person['person_id']]['bbox'] = _compute_yolo_bbox_from_keypoints(
            replaced_keypoints
        )

        result_image_path = source_person['image_path']
        result_label_path = label_path
        if output_dataset_path is not None:
            output_root = os.path.abspath(output_dataset_path)
            result_image_path = os.path.join(output_root, source_person['image_rel_path'])
            result_label_path = os.path.join(output_root, 'labels', source_person['label_name'])

        replacements.append(
            AxisAlignedKeypointReplacement(
                image_path=result_image_path,
                label_path=result_label_path,
                person_id=source_person['person_id'],
                original_keypoints=original_keypoints,
                replaced_keypoints=replaced_keypoints,
                changed_keypoint_indices=changed_keypoint_indices,
                shift_vectors=shift_vectors,
                direction_mode=str(direction_mode).lower(),
            )
        )

    _persist_replacement_dataset(
        yolo_dataset,
        label_entries_cache=label_entries_cache,
        output_dataset_path=output_dataset_path,
    )

    if isinstance(dataset, YOLOPoseDataset) and output_dataset_path is None:
        dataset._valid_pairs = dataset._validate_pairs()

    return replacements


def flatten_cvat_yolo_pose(
    input_root: str,
    output_root: str,
    copy_data_yaml: bool = True
) -> None:
    """Flatten nested CVAT YOLO pose dataset structure.
    
    Converts nested directory structure into flat images/ and labels/ folders.
    
    Args:
        input_root: Root directory of nested dataset
        output_root: Root directory for flattened output
        copy_data_yaml: Whether to copy data.yaml file
    """
    os.makedirs(output_root, exist_ok=True)

    out_images = os.path.join(output_root, "images")
    out_labels = os.path.join(output_root, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    label_index = {}

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".txt"):
                name = os.path.splitext(file)[0]
                label_index[name] = os.path.join(root, file)

    train_lines = []
    existing_names = set()

    image_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append((root, file))

    for root, file in tqdm(image_files, desc="Flattening dataset"):

        img_src = os.path.join(root, file)
        name, ext = os.path.splitext(file)

        new_name = file
        counter = 1

        while new_name in existing_names:
            new_name = f"{name}_{counter}{ext}"
            counter += 1

        existing_names.add(new_name)

        shutil.copy(img_src, os.path.join(out_images, new_name))

        if name in label_index:
            shutil.copy(
                label_index[name],
                os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
            )

        train_lines.append(f"images/{new_name}\n")

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(train_lines)

    if copy_data_yaml:
        for root, _, files in os.walk(input_root):
            if "data.yaml" in files:
                shutil.copy(
                    os.path.join(root, "data.yaml"),
                    os.path.join(output_root, "data.yaml")
                )
                break


def merge_yolo_pose_datasets(
    dataset1_root: str,
    dataset2_root: str,
    output_root: str
) -> None:
    """Merge two YOLO pose datasets.
    
    Combines images and labels from two datasets, handling filename conflicts.
    
    Args:
        dataset1_root: Root directory of first dataset
        dataset2_root: Root directory of second dataset
        output_root: Root directory for merged output
    """
    os.makedirs(output_root, exist_ok=True)
    out_images = os.path.join(output_root, "images")
    out_labels = os.path.join(output_root, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    merged_train_lines = []
    existing_names = set()

    def process_dataset(dataset_root: str, prefix: str) -> None:
        nonlocal merged_train_lines, existing_names

        images_path = os.path.join(dataset_root, "images")
        labels_path = os.path.join(dataset_root, "labels")
        train_txt_path = os.path.join(dataset_root, "train.txt")

        with open(train_txt_path, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Merging {prefix}", leave=False):
            img_rel_path = line.strip()
            img_name = os.path.basename(img_rel_path)

            name, ext = os.path.splitext(img_name)

            if img_name in existing_names:
                new_name = f"{prefix}_{img_name}"
            else:
                new_name = img_name

            existing_names.add(new_name)

            shutil.copy(
                os.path.join(images_path, img_name),
                os.path.join(out_images, new_name)
            )

            label_name = name + ".txt"
            src_label = os.path.join(labels_path, label_name)

            if os.path.exists(src_label):
                shutil.copy(
                    src_label,
                    os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
                )

            merged_train_lines.append(f"images/{new_name}\n")

    process_dataset(dataset1_root, "ds1")
    process_dataset(dataset2_root, "ds2")

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(merged_train_lines)

    data_yaml_1 = os.path.join(dataset1_root, "data.yaml")
    if os.path.exists(data_yaml_1):
        shutil.copy(data_yaml_1, os.path.join(output_root, "data.yaml"))


def split_yolo_pose_dataset(
    dataset_root: str,
    output_root: str,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
    dataset_path: Optional[str] = None,
    copy_mode: bool = False,
) -> None:
    """Split YOLO pose dataset into train and validation sets.
    
    Args:
        dataset_root: Root directory of input dataset
        output_root: Root directory for split output
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        dataset_path: Base path for data.yaml (defaults to output_root)
        copy_mode: If True, keep full train set and copy val subset from it
    """
    random.seed(seed)

    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")

    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)

    out_images_train = os.path.join(output_root, "images", "train")
    out_images_val = os.path.join(output_root, "images", "val")
    out_labels_train = os.path.join(output_root, "labels", "train")
    out_labels_val = os.path.join(output_root, "labels", "val")

    os.makedirs(out_images_train, exist_ok=True)
    os.makedirs(out_images_val, exist_ok=True)
    os.makedirs(out_labels_train, exist_ok=True)
    os.makedirs(out_labels_val, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - val_ratio))

    if copy_mode:
        val_count = len(image_files) - split_index
        train_files = image_files
        val_files = image_files[:val_count]
    else:
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

    def copy_files(file_list, img_out_dir: str, lbl_out_dir: str):
        txt_lines = []
        split_name = os.path.basename(img_out_dir)

        for img_name in tqdm(file_list, desc=f"Copying {split_name}", leave=False):
            name, ext = os.path.splitext(img_name)

            shutil.copy(
                os.path.join(images_root, img_name),
                os.path.join(img_out_dir, img_name)
            )

            label_path = os.path.join(labels_root, name + ".txt")
            if os.path.exists(label_path):
                shutil.copy(
                    label_path,
                    os.path.join(lbl_out_dir, name + ".txt")
                )

            txt_lines.append(f"images/{split_name}/{img_name}\n")

        return txt_lines

    train_lines = copy_files(train_files, out_images_train, out_labels_train)
    val_lines = copy_files(val_files, out_images_val, out_labels_val)

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(train_lines)

    with open(os.path.join(output_root, "val.txt"), "w") as f:
        f.writelines(val_lines)

    data_yaml_src = os.path.join(dataset_root, "data.yaml")
    data_yaml_dst = os.path.join(output_root, "data.yaml")

    yaml_content = ""

    if os.path.exists(data_yaml_src):
        with open(data_yaml_src, "r") as f:
            yaml_content = f.read()

    lines = yaml_content.splitlines()
    cleaned_lines = [
        line for line in lines
        if not line.strip().startswith(("path:", "train:", "val:"))
    ]

    if dataset_path is None:
        dataset_path = output_root

    cleaned_lines.insert(0, f"path: {dataset_path}")
    cleaned_lines.insert(1, "train: images/train")
    cleaned_lines.insert(2, "val: images/val")

    with open(data_yaml_dst, "w") as f:
        f.write("\n".join(cleaned_lines) + "\n")


def convert_yolo_pose_to_cvat(
    dataset_root: str,
    output_zip_dir: str
) -> None:
    """Convert YOLO pose dataset to CVAT format with ZIP archives.
    
    Args:
        dataset_root: Root directory of YOLO dataset
        output_zip_dir: Directory for output ZIP files
    """
    os.makedirs(output_zip_dir, exist_ok=True)

    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")
    train_txt_path = os.path.join(dataset_root, "train.txt")
    data_yaml_path = os.path.join(dataset_root, "data.yaml")

    nested_images_dir = os.path.join(labels_root, "train", "images")
    os.makedirs(nested_images_dir, exist_ok=True)

    # Copy images to nested structure
    image_files = [
        f for f in os.listdir(images_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in tqdm(image_files, desc="Copying images to nested structure"):
        shutil.copy(
            os.path.join(images_root, img_name),
            os.path.join(nested_images_dir, img_name)
        )

    # Update train.txt
    if not os.path.exists(train_txt_path):
        raise FileNotFoundError("train.txt not found")

    with open(train_txt_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        filename = os.path.basename(line.strip())
        updated_lines.append(f"data/images/train/{filename}\n")

    with open(train_txt_path, "w") as f:
        f.writelines(updated_lines)

    # ZIP images
    images_zip_path = os.path.join(output_zip_dir, "images.zip")

    with zipfile.ZipFile(images_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(images_root):
            for file in tqdm(files, desc="Zipping images", leave=False):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_root)
                zipf.write(full_path, rel_path)

    # ZIP annotations
    annotations_zip_path = os.path.join(output_zip_dir, "annotations.zip")

    with zipfile.ZipFile(annotations_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Labels
        for root, _, files in os.walk(labels_root):
            for file in tqdm(files, desc="Zipping labels", leave=False):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_root)
                zipf.write(full_path, rel_path)

        # data.yaml
        if os.path.exists(data_yaml_path):
            zipf.write(
                data_yaml_path,
                os.path.relpath(data_yaml_path, dataset_root)
            )

        # train.txt
        zipf.write(
            train_txt_path,
            os.path.relpath(train_txt_path, dataset_root)
        )


def extract_image_subset(
    images_path: str,
    output_dir: str,
    a: int,
    b: int
) -> None:
    """Extract images by index range from a folder.
    
    Copies images with indices [a, b] based on sorted folder listing.
    Useful for extracting specific subsets from large image collections.
    
    Args:
        images_path: Path to source folder with images
        output_dir: Path to destination folder
        a: Start index (inclusive)
        b: End index (inclusive)
        
    Returns:
        None. Prints number of images copied.
        
    Raises:
        IndexError: If index range is out of bounds
        ValueError: If a > b
        
    Example:
        >>> extract_image_subset(
        ...     images_path="./images",
        ...     output_dir="./subset",
        ...     a=0,
        ...     b=100
        ... )
         Skopiowano 101 obrazów do ./subset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of images
    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    n = len(image_files)

    if a < 0 or b >= n:
        raise IndexError(f"Index range must be in [0–{n-1}]")
    if a > b:
        raise ValueError("a must be <= b")

    selected = image_files[a:b+1]

    for img_file in selected:
        shutil.copy(
            os.path.join(images_path, img_file),
            os.path.join(output_dir, img_file)
        )

    print(f" Skopiowano {len(selected)} obrazów do {output_dir}")


def generate_train_txt_from_images(
    dataset_root: str,
    images_dir: str = "images",
    output_file: Optional[str] = None,
    recursive: bool = True,
) -> str:
    """Generate train.txt from image files stored under an images directory.

    Args:
        dataset_root: Root directory of the dataset.
        images_dir: Images directory path, relative to ``dataset_root`` or absolute.
        output_file: Output train.txt path. Defaults to ``dataset_root/train.txt``.
        recursive: Whether to scan nested image subdirectories.

    Returns:
        Absolute path to the generated train.txt file.

    Raises:
        FileNotFoundError: If the images directory does not exist.
        ValueError: If no images are found.
    """
    dataset_root_abs = os.path.abspath(dataset_root)
    images_root = images_dir if os.path.isabs(images_dir) else os.path.join(dataset_root_abs, images_dir)
    images_root = os.path.abspath(images_root)

    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    relative_image_paths: List[str] = []

    if recursive:
        for root, _, files in os.walk(images_root):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in image_extensions:
                    continue
                absolute_image_path = os.path.join(root, file_name)
                relative_image_path = os.path.relpath(absolute_image_path, dataset_root_abs).replace('\\', '/')
                relative_image_paths.append(relative_image_path)
    else:
        for file_name in os.listdir(images_root):
            absolute_image_path = os.path.join(images_root, file_name)
            if not os.path.isfile(absolute_image_path):
                continue
            extension = os.path.splitext(file_name)[1].lower()
            if extension not in image_extensions:
                continue
            relative_image_path = os.path.relpath(absolute_image_path, dataset_root_abs).replace('\\', '/')
            relative_image_paths.append(relative_image_path)

    relative_image_paths = sorted(set(relative_image_paths))
    if not relative_image_paths:
        raise ValueError(f"No image files found in: {images_root}")

    output_path = os.path.join(dataset_root_abs, "train.txt") if output_file is None else os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write("\n".join(relative_image_paths) + "\n")

    return output_path


def flatten_split_yolo_pose(
    dataset_root: str,
    splits: Optional[List[str]] = None,
    images_dir: str = "images",
    labels_dir: str = "labels",
    remove_split_dirs: bool = True,
    output_root: Optional[str] = None,
) -> str:
    """Flatten a split dataset (images/train, images/val, ...) into a single flat structure.

    Converts::

        dataset_root/
        ├── images/
        │   ├── train/   <-- all images moved up
        │   └── val/
        ├── labels/
        │   ├── train/   <-- all labels moved up
        │   └── val/
        ├── train.txt
        ├── val.txt
        └── data.yaml

    Into::

        dataset_root/  (or output_root/ if provided)
        ├── images/
        │   ├── img001.jpg
        │   └── ...
        ├── labels/
        │   ├── img001.txt
        │   └── ...
        ├── data.yaml
        └── train.txt   <-- regenerated from all images found

    Filename collisions across splits are resolved by prefixing the split name
    (e.g. ``val__img001.jpg``).

    Args:
        dataset_root: Root directory containing the split dataset.
        splits: List of split subdirectory names to merge. Defaults to any
            subdirectories found inside ``images_dir``.
        images_dir: Name of the images directory (relative to ``dataset_root``).
        labels_dir: Name of the labels directory (relative to ``dataset_root``).
        remove_split_dirs: If ``True``, delete empty split subdirectories after
            moving files. Ignored when ``output_root`` is set.
        output_root: Optional destination directory. When provided, files are
            *copied* (not moved) to this location and the source dataset is
            left untouched.

    Returns:
        Absolute path to the regenerated ``train.txt``.

    Raises:
        FileNotFoundError: If ``dataset_root`` or ``images_dir`` do not exist.
    """
    dataset_root_abs = os.path.abspath(dataset_root)
    images_root = os.path.join(dataset_root_abs, images_dir)
    labels_root = os.path.join(dataset_root_abs, labels_dir)

    if not os.path.isdir(dataset_root_abs):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root_abs}")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    # Auto-detect split subdirectories if not provided
    if splits is None:
        splits = [
            entry.name for entry in os.scandir(images_root)
            if entry.is_dir()
        ]

    if not splits:
        raise ValueError(f"No split subdirectories found in: {images_root}")

    # Determine destination root and file-transfer function
    copy_only = output_root is not None
    dest_root = os.path.abspath(output_root) if copy_only else dataset_root_abs
    dest_images_root = os.path.join(dest_root, images_dir)
    dest_labels_root = os.path.join(dest_root, labels_dir)
    os.makedirs(dest_images_root, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    # Track filenames already placed in the flat images dir to detect collisions
    placed_image_names: Set[str] = set()

    for split in splits:
        split_images_dir = os.path.join(images_root, split)
        split_labels_dir = os.path.join(labels_root, split) if os.path.isdir(labels_root) else None

        if not os.path.isdir(split_images_dir):
            continue

        for file_name in sorted(os.listdir(split_images_dir)):
            src_image = os.path.join(split_images_dir, file_name)
            if not os.path.isfile(src_image):
                continue
            if os.path.splitext(file_name)[1].lower() not in image_extensions:
                continue

            # Resolve collision: prefix split name if needed
            dest_name = file_name
            if dest_name in placed_image_names:
                dest_name = f"{split}__{file_name}"

            # Copy or move image
            dest_image_path = os.path.join(dest_images_root, dest_name)
            if copy_only:
                shutil.copy2(src_image, dest_image_path)
            else:
                shutil.move(src_image, dest_image_path)
            placed_image_names.add(dest_name)

            # Copy or move corresponding label if it exists
            if split_labels_dir is not None:
                stem = os.path.splitext(file_name)[0]
                src_label = os.path.join(split_labels_dir, stem + ".txt")
                if os.path.isfile(src_label):
                    dest_label_name = os.path.splitext(dest_name)[0] + ".txt"
                    os.makedirs(dest_labels_root, exist_ok=True)
                    dest_label_path = os.path.join(dest_labels_root, dest_label_name)
                    if copy_only:
                        shutil.copy2(src_label, dest_label_path)
                    else:
                        shutil.move(src_label, dest_label_path)

        # Remove now-empty split subdirectories (in-place mode only)
        if not copy_only and remove_split_dirs:
            for dir_path in [split_images_dir, os.path.join(labels_root, split) if split_labels_dir else None]:
                if dir_path and os.path.isdir(dir_path):
                    try:
                        os.rmdir(dir_path)  # only removes if empty
                    except OSError:
                        pass  # not empty — leave it

    # Copy data.yaml to output_root if applicable
    if copy_only:
        src_yaml = os.path.join(dataset_root_abs, "data.yaml")
        if os.path.isfile(src_yaml):
            shutil.copy2(src_yaml, os.path.join(dest_root, "data.yaml"))
    else:
        # Remove old split txt files (train.txt, val.txt, test.txt, etc.)
        for split in splits:
            old_txt = os.path.join(dataset_root_abs, f"{split}.txt")
            if os.path.isfile(old_txt):
                os.remove(old_txt)

    # Regenerate train.txt from all images now in the flat images dir
    train_txt_path = generate_train_txt_from_images(
        dataset_root=dest_root,
        images_dir=images_dir,
        recursive=False,
    )

    return train_txt_path
