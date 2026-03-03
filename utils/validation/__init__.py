"""Validation module for image analysis, duplicate detection, and model evaluation."""

from .helpers import (
    load_keypoints,
    load_keypoints_from_file,
    sequential_distances,
    draw_keypoints,
    draw_keypoints_with_patches,
)

from .duplicates import (
    generate_clip_embeddings,
    find_near_duplicates,
)

from .analysis import (
    generate_group_visualizations,
    save_groups_analysis,
    analyze_hidden_keypoints,
    export_groups_analysis_to_excel,
    train_distance_models,
    predict_distances,
    distance_matrix_from_keypoints,
    create_prediction_report,
)

from .evaluation import (
    compute_lbp_value,
    compute_lbp_value_variable_radius,
    extract_lbp_patch,
    compute_lbp_histogram,
    compute_lbp_histogram_variable_radius,
    compute_lbp_for_keypoints,
    compute_lbp_for_image,
    patch_to_binary_decimal,
    compute_lbp_binary_decimal,
    YOLOPoseDataset,
)

__all__ = [
    # helpers
    "load_keypoints",
    "load_keypoints_from_file",
    "sequential_distances",
    "draw_keypoints",
    "draw_keypoints_with_patches",
    # duplicates
    "generate_clip_embeddings",
    "find_near_duplicates",
    # analysis
    "generate_group_visualizations",
    "save_groups_analysis",
    "analyze_hidden_keypoints",
    "export_groups_analysis_to_excel",
    "train_distance_models",
    "predict_distances",
    "distance_matrix_from_keypoints",
    "create_prediction_report",
    # LBP feature extraction
    "compute_lbp_value",
    "compute_lbp_value_variable_radius",
    "extract_lbp_patch",
    "compute_lbp_histogram",
    "compute_lbp_histogram_variable_radius",
    "compute_lbp_for_keypoints",
    "compute_lbp_for_image",
    "patch_to_binary_decimal",
    "compute_lbp_binary_decimal",
    "YOLOPoseDataset",
]
