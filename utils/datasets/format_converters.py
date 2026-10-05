"""Converters from custom keypoint dataset formats to YOLO Pose format."""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from .dataset_formats import FORMAT_REGISTRY, KeypointAnnotation
from .operations import generate_train_txt_from_images


def _ensure_yolo_bbox(annotation: KeypointAnnotation) -> List[float]:
    if annotation.bbox_xywh is not None:
        return [float(v) for v in annotation.bbox_xywh]

    keypoints = np.asarray(annotation.keypoints, dtype=float)
    if keypoints.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    coords = keypoints[:, :2]
    if keypoints.shape[1] >= 3:
        visible_mask = keypoints[:, 2] > 0
        if np.any(visible_mask):
            coords = coords[visible_mask]

    if coords.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

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


def _annotation_to_yolo_line(annotation: KeypointAnnotation) -> str:
    class_id = int(annotation.class_id)
    bbox = _ensure_yolo_bbox(annotation)

    keypoints = np.asarray(annotation.keypoints, dtype=float)
    if keypoints.size == 0:
        return " ".join([str(class_id)] + [f"{v:.6f}" for v in bbox])

    if keypoints.ndim != 2 or keypoints.shape[1] not in (2, 3):
        raise ValueError(f"Invalid keypoints shape {keypoints.shape}; expected (N,2) or (N,3)")

    coords = np.clip(keypoints[:, :2], 0.0, 1.0)
    if keypoints.shape[1] == 3:
        visibility = keypoints[:, 2:3]
    else:
        visibility = np.full((coords.shape[0], 1), 2.0, dtype=float)

    keypoints_xyv = np.hstack([coords, visibility]).reshape(-1)

    serialized = [str(class_id)]
    serialized.extend(f"{float(v):.6f}" for v in bbox)
    serialized.extend(f"{float(v):.6f}" for v in keypoints_xyv)
    return " ".join(serialized)


def convert_registered_format_to_yolo_pose(
    format_name: str,
    input_root: str,
    output_root: str,
    split_name: str = "train",
    copy_images: bool = True,
    generate_split_file: bool = True,
) -> Dict[str, object]:
    """Convert a registered format to YOLO Pose directory structure.

    Output layout:
      output_root/images/...\n
      output_root/labels/...\n
      output_root/train.txt (optional)
    """
    format_impl = FORMAT_REGISTRY.get(format_name)

    input_root_abs = os.path.abspath(input_root)
    output_root_abs = os.path.abspath(output_root)
    images_root = os.path.join(output_root_abs, "images")
    labels_root = os.path.join(output_root_abs, "labels")
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    converted_images = 0
    converted_annotations = 0
    skipped_missing_images = 0

    for sample in format_impl.iter_samples(input_root_abs):
        rel_image_path = sample.relative_image_path.replace('\\', '/')
        source_image_path = sample.image_path

        if not os.path.isfile(source_image_path):
            skipped_missing_images += 1
            continue

        destination_image_path = os.path.join(output_root_abs, rel_image_path)
        destination_label_path = os.path.join(
            labels_root,
            os.path.splitext(rel_image_path)[0] + ".txt",
        )

        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(destination_label_path), exist_ok=True)

        if copy_images:
            shutil.copy2(source_image_path, destination_image_path)

        lines: List[str] = []
        for ann in sample.annotations:
            lines.append(_annotation_to_yolo_line(ann))
            converted_annotations += 1

        with open(destination_label_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))

        converted_images += 1

    split_path: Optional[str] = None
    if generate_split_file:
        split_path = generate_train_txt_from_images(
            dataset_root=output_root_abs,
            images_dir="images",
            output_file=os.path.join(output_root_abs, f"{split_name}.txt"),
            recursive=True,
        )

    return {
        "format": format_name,
        "input_root": input_root_abs,
        "output_root": output_root_abs,
        "split_file": split_path,
        "converted_images": converted_images,
        "converted_annotations": converted_annotations,
        "skipped_missing_images": skipped_missing_images,
    }


def convert_coco_keypoints_with_ultralytics(
    coco_annotations_dir: str,
    output_root: str,
    copy_images_from: Optional[str] = None,
    split_name: str = "train",
) -> Dict[str, object]:
    """Use Ultralytics COCO converter for keypoints when available.

    This wraps Ultralytics conversion utility and then optionally copies images
    into ``output_root/images`` and regenerates ``split_name`` txt file.
    """
    try:
        from ultralytics.data.converter import convert_coco
    except Exception as exc:
        raise ImportError(
            "Ultralytics converter is unavailable. Install ultralytics to use this path."
        ) from exc

    coco_annotations_dir_abs = os.path.abspath(coco_annotations_dir)
    output_root_abs = os.path.abspath(output_root)

    os.makedirs(output_root_abs, exist_ok=True)
    convert_coco(
        labels_dir=coco_annotations_dir_abs,
        save_dir=output_root_abs,
        use_segments=False,
        use_keypoints=True,
    )

    copied_images = 0
    if copy_images_from is not None:
        images_src_abs = os.path.abspath(copy_images_from)
        images_dst_abs = os.path.join(output_root_abs, "images")
        os.makedirs(images_dst_abs, exist_ok=True)

        for root, _, files in os.walk(images_src_abs):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    continue
                src = os.path.join(root, file_name)
                rel = os.path.relpath(src, images_src_abs)
                dst = os.path.join(images_dst_abs, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied_images += 1

    split_path = generate_train_txt_from_images(
        dataset_root=output_root_abs,
        images_dir="images",
        output_file=os.path.join(output_root_abs, f"{split_name}.txt"),
        recursive=True,
    )

    return {
        "converter": "ultralytics.convert_coco",
        "annotations_dir": coco_annotations_dir_abs,
        "output_root": output_root_abs,
        "copied_images": copied_images,
        "split_file": split_path,
    }
