"""Dataset operations module for YOLO pose dataset manipulation."""

from .yolo_pose_dataset import YOLOPoseImage, YOLOPoseDataset
from .operations import (
    AnglePreservingKeypointReplacement,
    AxisAlignedKeypointReplacement,
    RigidKeypointReplacement,
    flatten_cvat_yolo_pose,
    replace_angle_preserving_keypoints,
    replace_axis_aligned_keypoints,
    replace_rigid_keypoints,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    export_random_dataset_subset,
    export_dataset_partitions,
    remove_duplicate_samples_from_dataset,
    remove_samples_by_image_paths,
    generate_train_txt_from_images,
    flatten_split_yolo_pose,
    save_results_console_output,
    extract_image_subset,
)

__all__ = [
    # Dataset classes
    "AnglePreservingKeypointReplacement",
    "AxisAlignedKeypointReplacement",
    "RigidKeypointReplacement",
    "YOLOPoseImage",
    "YOLOPoseDataset",
    # Operations
    "flatten_cvat_yolo_pose",
    "replace_angle_preserving_keypoints",
    "replace_axis_aligned_keypoints",
    "replace_rigid_keypoints",
    "merge_yolo_pose_datasets",
    "split_yolo_pose_dataset",
    "convert_yolo_pose_to_cvat",
    "export_random_dataset_subset",
    "export_dataset_partitions",
    "remove_duplicate_samples_from_dataset",
    "remove_samples_by_image_paths",
    "generate_train_txt_from_images",
    "flatten_split_yolo_pose",
    "save_results_console_output",
    "extract_image_subset",
]
