"""Helper utilities for keypoint manipulation and visualization."""

from typing import Tuple, List
import numpy as np
import cv2


def load_keypoints(label_path: str) -> np.ndarray:
    """Load keypoints from a single YOLO pose label file.
    
    Reads the first line of a label file and extracts normalized keypoint coordinates
    (x, y pairs), skipping the bounding box values.
    
    Args:
        label_path: Path to .txt label file in YOLO pose format
                   Format: class_id bbox_cx bbox_cy bbox_w bbox_h x1 y1 vis1 x2 y2 vis2 ...
        
    Returns:
        Array of shape (N, 2) with normalized keypoint coordinates [x, y] in range [0, 1]
        
    Example:
        >>> keypoints = load_keypoints("image_001.txt")
        >>> keypoints.shape
        (9, 2)  # 9 keypoints
        >>> keypoints[0]
        array([0.5, 0.4])  # Normalized coordinates
    """
    with open(label_path, "r") as f:
        line = f.readline().strip().split()

    values = list(map(float, line[5:]))

    keypoints = []
    for i in range(0, len(values), 3):
        keypoints.append([values[i], values[i + 1]])

    return np.array(keypoints)


def load_keypoints_from_file(dataset_file: str, labels_dir: str = None) -> List[np.ndarray]:
    """Load all keypoints from all images in a YOLO pose dataset file.
    
    Reads a dataset file (e.g., train.txt) containing image paths, then loads keypoints
    from all corresponding label files. If a label file contains multiple objects (lines),
    each object's keypoints are returned as a separate array.
    
    Args:
        dataset_file: Path to dataset file with image paths (one per line).
                     Image paths should be relative to the dataset directory.
                     Example: train.txt containing "images/img_001.jpg", "images/img_002.jpg"
        labels_dir: Path to labels directory. If None, inferred from dataset_file parent.
                   Defaults to: os.path.dirname(dataset_file)/labels/
        
    Returns:
        List of keypoint arrays. One array per object/line across all label files.
        Each array has shape (N_keypoints, 2) with normalized [x, y] coordinates.
        
    Example:
        >>> # train.txt contains 2 images, label files contain 4 objects total
        >>> keypoints_list = load_keypoints_from_file("train.txt")
        >>> len(keypoints_list)
        4  # 4 objects total
        >>> keypoints_list[0].shape
        (9, 2)  # First object has 9 keypoints
        
    Note:
        - Skips missing label files silently (image_path in train.txt with no .txt label)
        - Does NOT return image paths or image correspondence
        - For per-image processing, use compute_lbp_for_dataset() instead
    """
    import os
    
    keypoints_list = []
    
    # Infer labels directory if not provided
    if labels_dir is None:
        dataset_dir = os.path.dirname(dataset_file)
        labels_dir = os.path.join(dataset_dir, "labels")
    
    with open(dataset_file, "r") as f:
        for image_path in f.readlines():
            image_path = image_path.strip()
            if not image_path:
                continue
            
            # Convert image path to label path
            label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            
            if not os.path.exists(label_path):
                continue
            
            # Load keypoints from label file
            with open(label_path, "r") as lf:
                for line in lf.readlines():
                    line = line.strip().split()
                    if not line:
                        continue
                    
                    values = list(map(float, line[5:]))
                    
                    keypoints = []
                    for i in range(0, len(values), 3):
                        keypoints.append([values[i], values[i + 1]])
                    
                    keypoints_list.append(np.array(keypoints))
    
    return keypoints_list


def sequential_distances(keypoints: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances between consecutive keypoints.
    
    Computes the distance from keypoint[i] to keypoint[i+1] for all pairs.
    Useful for analyzing motion between consecutive keypoints in a sequence.
    
    Args:
        keypoints: Array of shape (N, 2) with keypoint coordinates [x, y]
        
    Returns:
        Array of shape (N-1,) with distances between consecutive keypoints.
        Returns empty array if fewer than 2 keypoints provided.
        
    Example:
        >>> keypoints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        >>> distances = sequential_distances(keypoints)
        >>> distances
        array([1., 1.])  # Distance 0→1 is 1.0, distance 1→2 is 1.0
    """
    return np.array([
        np.linalg.norm(keypoints[i] - keypoints[i + 1])
        for i in range(len(keypoints) - 1)
    ])


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw keypoints as circles on an image.
    
    Overlays circles at keypoint locations. Modifies image in-place and returns it.
    Handles normalized coordinates [0, 1] by converting to pixel coordinates.
    
    Args:
        image: Input image as numpy array with shape (H, W) or (H, W, 3)
        keypoints: Array of shape (N, 2) with normalized keypoint coordinates [0, 1]
                  where 0.0 = left/top and 1.0 = right/bottom
        color: RGB color tuple (R, G, B) in range [0, 255] for drawing circles
               Example: (0, 255, 0) for green, (255, 0, 0) for red
        
    Returns:
        Modified image with keypoints drawn as circles (radius=4, filled).
        Returns a copy of the image with overlaid circles.
        
    Example:
        >>> import cv2
        >>> image = cv2.imread("image.jpg")
        >>> keypoints = np.array([[0.5, 0.5], [0.3, 0.7]])  # Normalized coords
        >>> result = draw_keypoints(image, keypoints, (0, 255, 0))  # Green circles
        >>> cv2.imshow("With keypoints", result)
    """
    h, w = image.shape[:2]
    for (x, y) in keypoints:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(image, (px, py), 4, color, -1)
    return image


def draw_keypoints_with_patches(image: np.ndarray, 
                                keypoints: np.ndarray, 
                                patch_size: int,
                                keypoint_color: Tuple[int, int, int] = (0, 255, 0),
                                patch_color: Tuple[int, int, int] = (255, 0, 0),
                                thickness: int = 2) -> np.ndarray:
    """
    Draw keypoints and surrounding patch rectangles on an image.
    
    For each keypoint, draws:
    1. A rectangle (patch_size × patch_size) centered on the keypoint
    2. A circle at the keypoint center
    
    Args:
        image: Input image as numpy array with shape (H, W) or (H, W, 3)
        keypoints: Array of shape (N, 2) with normalized keypoint coordinates [0, 1]
                  where 0.0 = left/top and 1.0 = right/bottom
        patch_size: Size of square patch to draw around each keypoint (in pixels)
        keypoint_color: RGB color tuple (R, G, B) for keypoint circles (default: green)
        patch_color: RGB color tuple for patch rectangles (default: blue)
        thickness: Line thickness for rectangle borders. If 0, rectangle is filled.
                  Negative value (e.g., -1) fills the rectangle.
        
    Returns:
        Modified image with keypoints and patches drawn.
        
    Example:
        >>> import cv2
        >>> image = cv2.imread("image.jpg")
        >>> keypoints = np.array([[0.5, 0.5], [0.3, 0.7]])  # Normalized coords
        >>> result = draw_keypoints_with_patches(
        ...     image, keypoints, patch_size=32,
        ...     keypoint_color=(0, 255, 0),  # Green circles
        ...     patch_color=(255, 0, 0)       # Blue rectangles
        ... )
        >>> cv2.imshow("With patches", result)
    """
    h, w = image.shape[:2]
    half_patch = patch_size // 2
    
    for (x, y) in keypoints:
        # Convert normalized to pixel coordinates
        px = int(x * w)
        py = int(y * h)
        
        # Draw patch rectangle (centered on keypoint)
        x1 = max(0, px - half_patch)
        y1 = max(0, py - half_patch)
        x2 = min(w - 1, px + half_patch)
        y2 = min(h - 1, py + half_patch)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), patch_color, thickness)
        
        # Draw keypoint circle at center
        cv2.circle(image, (px, py), 4, keypoint_color, -1)
    
    return image
