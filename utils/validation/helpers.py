"""
DEPRECATED: helpers.py is deprecated. Use class methods instead:

YPImageValidation class methods:
- load_keypoints(label_path, include_visibility=True) - classmethod
- compute_sequential_distances(keypoints, visibility_threshold=0.5) - staticmethod
- draw_keypoints_on_image(image, keypoints, color) - staticmethod

YPSetValidation class methods:
- load_all_keypoints(dataset_file, labels_dir, split) - staticmethod

This file is kept for reference only and will be removed in future versions.
All functions have been moved to YPImageValidation and YPSetValidation classes.
"""