"""Validation module for image analysis, duplicate detection, and model evaluation."""

from importlib import import_module
from typing import Any, Dict, Tuple


_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    
    "parse_results_txt": (".weryfikacje.yp_validation_test", "parse_results_txt"),
    "parse_results_txt_line": (".weryfikacje.yp_validation_test", "parse_results_txt_line"),
    "parse_distance_anomaly_report": (".weryfikacje.yp_validation_test", "parse_distance_anomaly_report"),
    "compare_results_txt": (".weryfikacje.yp_validation_test", "compare_results_txt"),
    "compare_results_txt_entries": (".weryfikacje.yp_validation_test", "compare_results_txt_entries"),
    "compare_results_txt_to_distance_anomalies": (".weryfikacje.yp_validation_test", "compare_results_txt_to_distance_anomalies"),
    "compare_results_txt_to_distance_anomalies_with_outside_keypoints": (".weryfikacje.yp_validation_test", "compare_results_txt_to_distance_anomalies_with_outside_keypoints"),
    "copy_missing_person_labels_from_results": (".weryfikacje.yp_validation_test", "copy_missing_person_labels_from_results"),
    "save_distance_anomaly_report": (".weryfikacje.yp_validation_test", "save_distance_anomaly_report"),
    "summarize_results_txt": (".weryfikacje.yp_validation_test", "summarize_results_txt"),
    "summarize_results_txt_entries": (".weryfikacje.yp_validation_test", "summarize_results_txt_entries"),
    "YPValidation_Test": (".weryfikacje.yp_validation_test", "YPValidation_Test"),
    "YOLOPoseImage": ("..datasets.yolo_pose_dataset", "YOLOPoseImage"),
    "YOLOPoseDataset": ("..datasets.yolo_pose_dataset", "YOLOPoseDataset"),
    "YPImageValidation": (".image_validation", "YPImageValidation"),
    "YPSetValidation": (".set_validation", "YPSetValidation"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


__all__ = [
    # analysis (removed)
    "parse_results_txt",
    "parse_results_txt_line",
    "parse_distance_anomaly_report",
    "compare_results_txt",
    "compare_results_txt_entries",
    "compare_results_txt_to_distance_anomalies",
    "compare_results_txt_to_distance_anomalies_with_outside_keypoints",
    "copy_missing_person_labels_from_results",
    "save_distance_anomaly_report",
    "summarize_results_txt",
    "summarize_results_txt_entries",
    # dataset classes
    "YOLOPoseImage",
    "YOLOPoseDataset",
    # validation classes
    "YPImageValidation",
    "YPSetValidation",
    "YPValidation_Test",
]
