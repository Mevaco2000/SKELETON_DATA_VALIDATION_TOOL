"""
Professional utilities package for YOLO pose dataset validation and manipulation.

Provides tools for:
- Keypoint analysis and visualization
- Duplicate detection using CLIP embeddings
- Dataset statistics and Excel reporting
- YOLO model training and evaluation
- Dataset operations (flatten, merge, split, convert)
"""

__version__ = "0.1.0"
__author__ = "Rafal Wysocki, 2026"

# Import from validation module
from .validation import (
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel,
    YOLOPoseDataset,
    YPImageValidation,
    YPSetValidation,
)

# Import from datasets module
from .datasets import (
    YOLOPoseImage,
    YOLOPoseDataset,
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset,
)

# Import config for users who need constants
from .config import MODEL_REGISTRY

__all__ = [
    # Validation - analysis
    "generate_group_visualizations",
    "save_groups_analysis",
    "analyze_hidden_keypoints",
    "export_groups_analysis_to_excel",
    # Validation - classes
    "YPImageValidation",
    "YPSetValidation",
    # Datasets - classes
    "YOLOPoseImage",
    "YOLOPoseDataset",
    # Datasets - operations
    "flatten_cvat_yolo_pose",
    "merge_yolo_pose_datasets",
    "split_yolo_pose_dataset",
    "convert_yolo_pose_to_cvat",
    "extract_image_subset",
    # Config
    "MODEL_REGISTRY",
]
