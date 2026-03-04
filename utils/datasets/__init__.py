"""Dataset operations module for YOLO pose dataset manipulation."""

from .yolo_pose_dataset import YOLOPoseImage, YOLOPoseDataset
from .operations import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset,
)

__all__ = [
    # Dataset classes
    "YOLOPoseImage",
    "YOLOPoseDataset",
    # Operations
    "flatten_cvat_yolo_pose",
    "merge_yolo_pose_datasets",
    "split_yolo_pose_dataset",
    "convert_yolo_pose_to_cvat",
    "extract_image_subset",
]
