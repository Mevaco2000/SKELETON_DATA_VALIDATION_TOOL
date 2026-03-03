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
    load_keypoints,
    sequential_distances,
    draw_keypoints,
    draw_keypoints_with_patches,
    generate_clip_embeddings,
    find_near_duplicates,
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel,
    compute_lbp_value,
    compute_lbp_for_keypoints,
    compute_lbp_for_image,
    patch_to_binary_decimal,
    compute_lbp_binary_decimal,
    YOLOPoseDataset,
)

# Import from datasets module
from .datasets import (
    flatten_cvat_yolo_pose,
    merge_yolo_pose_datasets,
    split_yolo_pose_dataset,
    convert_yolo_pose_to_cvat,
    extract_image_subset,
)

# Import config for users who need constants
from .config import MODEL_REGISTRY

__all__ = [
    # Validation - keypoint utilities
    "load_keypoints",
    "sequential_distances",
    "draw_keypoints",
    "draw_keypoints_with_patches",
    # Validation - duplicate detection
    "generate_clip_embeddings",
    "find_near_duplicates",
    # Validation - analysis
    "generate_group_visualizations",
    "save_groups_analysis",
    "analyze_hidden_keypoints",
    "export_groups_analysis_to_excel",
    # Validation - LBP features
    "compute_lbp_value",
    "compute_lbp_for_keypoints",
    "compute_lbp_for_image",
    "patch_to_binary_decimal",
    "compute_lbp_binary_decimal",
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
