"""Single image LBP validation methods."""

from typing import Union
import numpy as np
import cv2


class YPImageValidation:
    """
    Validation methods for a single image with annotations.
    
    Handles LBP computation, histograms, and binary decimal representations
    for individual images with keypoint annotations.
    
    Use this class for per-image or per-keypoint feature extraction.
    Use YPSetValidation for dataset-level operations.
    
    USAGE PATTERNS:
    
    Example 1 - Compute LBP for a single image:
        >>> from github.utils.validation import YPImageValidation
        >>> import cv2
        >>> 
        >>> image = cv2.imread('image.jpg')
        >>> keypoints = np.array([[100, 50], [150, 200]])  # (N, 2)
        >>> 
        >>> validator = YPImageValidation(image, keypoints)
        >>> hist = validator.compute_lbp_histograms(patch_size=5)  # (N, 256)
        >>> decimals = validator.compute_lbp_decimals(patch_size=7)  # (N,)
    
    Example 2 - Process individual keypoints:
        >>> validator = YPImageValidation(image, keypoints)
        >>> for i, (x, y) in enumerate(keypoints):
        ...     hist = validator.compute_lbp_histogram_single(int(x), int(y))
        ...     decimal = validator.compute_lbp_decimal_single(int(x), int(y))
    
    Example 3 - Visualize single image:
        >>> validator = YPImageValidation(image, keypoints)
        >>> result = validator.visualize(patch_size=5)
    """
    
    def __init__(self, image: Union[np.ndarray, str], keypoints: np.ndarray):
        """
        Initialize image validator.
        
        Args:
            image: Image as numpy array (H x W or H x W x 3) or path to image file
            keypoints: Array of keypoints with shape (N, 2) with [x, y] OR (N, 3) with [x, y, visibility].
                      If (N, 3), visibility is checked when filtering keypoints.
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.image = image
        self.keypoints = np.asarray(keypoints)
        self.h, self.w = image.shape[:2]
    
    def sequential_distances(self, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Calculate Euclidean distances between consecutive visible keypoints.
        
        Computes the distance from keypoint[i] to keypoint[i+1] for consecutive visible pairs.
        Skips distances involving hidden/occluded keypoints (visibility < threshold).
        
        Args:
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
        
        Returns:
            Array of distances between consecutive visible keypoints.
            Returns empty array if fewer than 2 visible keypoints.
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> distances = validator.sequential_distances(visibility_threshold=0.5)
            >>> print(distances)  # Array of distances between consecutive keypoints
        """
        return YPImageValidation.compute_sequential_distances(
            self.keypoints, visibility_threshold=visibility_threshold
        )
    
    @staticmethod
    def compute_sequential_distances(keypoints: np.ndarray, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Calculate Euclidean distances between consecutive visible keypoints (static).
        
        Static method for computing sequential distances without creating an instance.
        
        Args:
            keypoints: Array of shape (N, 2) with [x, y] OR (N, 3) with [x, y, visibility].
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
        
        Returns:
            Array of distances between consecutive visible keypoints.
            
        Example:
            >>> kpts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.3], [1.0, 1.0, 1.0]])
            >>> distances = YPImageValidation.compute_sequential_distances(kpts, visibility_threshold=0.5)
        """
        keypoints = np.asarray(keypoints)
        coords = keypoints[:, :2]
        
        if keypoints.shape[1] >= 3:
            visibility = keypoints[:, 2]
            visible_mask = visibility >= visibility_threshold
        else:
            visible_mask = np.ones(len(keypoints), dtype=bool)
        
        distances = []
        for i in range(len(keypoints) - 1):
            if visible_mask[i] and visible_mask[i + 1]:
                dist = np.linalg.norm(coords[i] - coords[i + 1])
                distances.append(dist)
        
        return np.array(distances) if distances else np.array([])
    
    @staticmethod
    def _compute_lbp_value(neighborhood: np.ndarray) -> int:
        """Compute LBP value for 3x3 neighborhood."""
        center = neighborhood[1, 1]
        binary_string = ""
        neighbors = [
            neighborhood[0, 0], neighborhood[0, 1], neighborhood[0, 2],
            neighborhood[1, 2], neighborhood[2, 2], neighborhood[2, 1],
            neighborhood[2, 0], neighborhood[1, 0]
        ]
        for neighbor in neighbors:
            binary_string += "1" if neighbor >= center else "0"
        return int(binary_string, 2)
    
    @staticmethod
    def _patch_to_binary_decimal(patch: np.ndarray, center_value: float = None) -> dict:
        """Convert patch to binary decimal representation."""
        if center_value is None:
            h, w = patch.shape
            center_idx_h, center_idx_w = h // 2, w // 2
            center_value = patch[center_idx_h, center_idx_w]
        
        binary_matrix = (patch >= center_value).astype(np.uint8)
        binary_string = ''.join(binary_matrix.flatten().astype(str))
        decimal_value = int(binary_string, 2)
        
        return {
            'binary_matrix': binary_matrix,
            'binary_string': binary_string,
            'decimal': decimal_value,
            'patch_shape': patch.shape
        }
    
    def _compute_lbp_histogram_single_internal(self, x: int, y: int, patch_size: int = 5) -> np.ndarray:
        """Internal method to compute LBP histogram for single keypoint."""
        half_size = patch_size // 2
        
        x_start = max(0, int(x) - half_size)
        x_end = min(self.w, int(x) + half_size + 1)
        y_start = max(0, int(y) - half_size)
        y_end = min(self.h, int(y) + half_size + 1)
        
        patch = self.image[y_start:y_end, x_start:x_end].astype(np.uint8)
        
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            patch = np.pad(patch, ((half_size, half_size), (half_size, half_size)), mode='edge')
        
        lbp_map = np.zeros(patch.shape, dtype=np.uint8)
        for py in range(1, patch.shape[0] - 1):
            for px in range(1, patch.shape[1] - 1):
                lbp_map[py, px] = self._compute_lbp_value(patch[py-1:py+2, px-1:px+2])
        
        hist, _ = np.histogram(lbp_map, bins=256, range=(0, 256))
        return hist.astype(np.float32)
    
    def compute_lbp_histogram_single(self, x: int, y: int, patch_size: int = 5) -> np.ndarray:
        """
        Compute LBP histogram for single keypoint.
        
        Args:
            x: X coordinate of keypoint
            y: Y coordinate of keypoint
            patch_size: Size of patch around keypoint
            
        Returns:
            LBP histogram (256,)
        """
        return self._compute_lbp_histogram_single_internal(x, y, patch_size)
    
    def compute_lbp_decimal_single(self, x: int, y: int, patch_size: int = 5) -> np.uint64:
        """
        Compute binary decimal representation for single keypoint.
        
        Args:
            x: X coordinate of keypoint
            y: Y coordinate of keypoint
            patch_size: Size of patch around keypoint
            
        Returns:
            Binary decimal value (uint64)
        """
        half_size = patch_size // 2
        
        x_start = max(0, int(x) - half_size)
        x_end = min(self.w, int(x) + half_size + 1)
        y_start = max(0, int(y) - half_size)
        y_end = min(self.h, int(y) + half_size + 1)
        
        patch = self.image[y_start:y_end, x_start:x_end].astype(np.uint8)
        
        if patch.size == 0:
            return np.uint64(0)
        
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='edge')
        
        result = self._patch_to_binary_decimal(patch)
        return np.uint64(result['decimal'])
    
    def compute_lbp_histograms(self, patch_size: int = 5, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Compute LBP histograms for all visible keypoints in the image.
        
        Skips keypoints with visibility below threshold (hidden/occluded keypoints).
        
        Args:
            patch_size: Size of patch around each keypoint
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Array of shape (N_visible, 256) - one histogram per visible keypoint.
            N_visible <= N if keypoints include visibility information.
        """
        if self.keypoints.size == 0:
            return np.zeros((0, 256), dtype=np.float32)
        
        # Filter by visibility if available
        if self.keypoints.shape[1] >= 3:
            visibility = self.keypoints[:, 2]
            visible_indices = np.where(visibility >= visibility_threshold)[0]
            coords = self.keypoints[visible_indices, :2]
        else:
            coords = self.keypoints
        
        lbp_features = np.zeros((len(coords), 256), dtype=np.float32)
        for i, (x, y) in enumerate(coords):
            lbp_features[i] = self.compute_lbp_histogram_single(int(x), int(y), patch_size)
        
        return lbp_features
    
    def compute_lbp_histograms_variable_radius(self, patch_size: int = 5, 
                                               radius: int = 1, n_points: int = 8,
                                               visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Compute LBP histograms with variable radius for all visible keypoints.
        
        Skips keypoints with visibility below threshold (hidden/occluded keypoints).
        
        Args:
            patch_size: Size of patch around each keypoint
            radius: Radius of circular neighborhood
            n_points: Number of sampling points
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Array of shape (N_visible, bins) - one histogram per visible keypoint.
            N_visible <= N if keypoints include visibility information.
        """
        if self.keypoints.size == 0:
            return np.zeros((0, 256), dtype=np.float32)
        
        # Filter by visibility if available
        if self.keypoints.shape[1] >= 3:
            visibility = self.keypoints[:, 2]
            visible_indices = np.where(visibility >= visibility_threshold)[0]
            coords = self.keypoints[visible_indices, :2]
        else:
            coords = self.keypoints
        
        max_val = 2 ** n_points
        bins = 256
        lbp_features = np.zeros((len(coords), bins), dtype=np.float32)
        
        for i, (x, y) in enumerate(coords):
            x_int, y_int = int(x), int(y)
            lbp_features[i] = self._compute_lbp_histogram_variable_radius_single(
                x_int, y_int, radius, n_points, patch_size, bins
            )
        
        return lbp_features
    
    @staticmethod
    def _compute_lbp_value_variable_radius(image: np.ndarray, y: int, x: int,
                                          radius: int = 1, n_points: int = 8) -> int:
        """Compute LBP value using circular sampling at variable radius."""
        h, w = image.shape[:2]
        center_val = image[int(y), int(x)]
        binary_string = ""
        
        for i in range(n_points):
            angle = 2.0 * np.pi * i / n_points
            px = int(x + radius * np.cos(angle))
            py = int(y - radius * np.sin(angle))
            
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)
            
            neighbor_val = image[py, px]
            binary_string += "1" if neighbor_val >= center_val else "0"
        
        return int(binary_string, 2)
    
    def _compute_lbp_histogram_variable_radius_single(self, x: int, y: int,
                                                      radius: int, n_points: int,
                                                      patch_size: int, bins: int) -> np.ndarray:
        """Internal method for variable radius LBP histogram."""
        half_size = patch_size // 2
        
        x_start = max(0, x - half_size)
        x_end = min(self.w, x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(self.h, y + half_size + 1)
        
        patch = self.image[y_start:y_end, x_start:x_end].astype(np.uint8)
        
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            patch = np.pad(patch, ((half_size, half_size), (half_size, half_size)), mode='edge')
        
        lbp_map = np.zeros(patch.shape, dtype=np.uint32)
        py_offset = y_start
        px_offset = x_start
        
        for py in range(patch.shape[0]):
            for px in range(patch.shape[1]):
                img_y = py_offset + py
                img_x = px_offset + px
                if 0 <= img_y < self.h and 0 <= img_x < self.w:
                    lbp_map[py, px] = self._compute_lbp_value_variable_radius(
                        self.image, img_y, img_x, radius, n_points
                    )
        
        max_val = 2 ** n_points
        hist, _ = np.histogram(lbp_map, bins=bins, range=(0, min(bins, max_val)))
        return hist.astype(np.float32)
    
    def compute_lbp_decimals(self, patch_size: int = 5, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Compute binary decimal representations for all visible keypoints in the image.
        
        Skips keypoints with visibility below threshold (hidden/occluded keypoints).
        
        Args:
            patch_size: Size of patch around each keypoint
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Array of shape (N_visible,) - uint64 values, one per visible keypoint.
            N_visible <= N if keypoints include visibility information.
        """
        if self.keypoints.size == 0:
            return np.array([], dtype=np.uint64)
        
        # Filter by visibility if available
        if self.keypoints.shape[1] >= 3:
            visibility = self.keypoints[:, 2]
            visible_indices = np.where(visibility >= visibility_threshold)[0]
            coords = self.keypoints[visible_indices, :2]
        else:
            coords = self.keypoints
        
        decimal_values = []
        for x, y in coords:
            decimal_values.append(self.compute_lbp_decimal_single(x, y, patch_size))
        
        return np.array(decimal_values, dtype=np.uint64)
    
    def draw_keypoints(self, color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        Draw keypoints as circles on image.
        
        Args:
            color: RGB color tuple (R, G, B) in range [0, 255].
                  Default (0, 255, 0) for green.
        
        Returns:
            Image with keypoints drawn as circles
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> result = validator.draw_keypoints(color=(0, 255, 0))
            >>> cv2.imshow("With keypoints", result)
        """
        return YPImageValidation.draw_keypoints_on_image(
            self.image, self.keypoints, color=color
        )
    
    @staticmethod
    def draw_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray, 
                               color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        Draw keypoints as circles on image (static method).
        
        Static method for drawing keypoints without creating an instance.
        
        Args:
            image: Image as numpy array (H x W or H x W x 3)
            keypoints: Array of keypoints (N, 2) with normalized [x, y] coordinates
            color: RGB color tuple (R, G, B) in range [0, 255]
        
        Returns:
            Image with keypoints drawn
            
        Example:
            >>> result = YPImageValidation.draw_keypoints_on_image(image, keypoints, color=(0, 255, 0))
        """
        import copy
        result = copy.copy(image)
        h, w = image.shape[:2] if len(image.shape) >= 2 else (image.shape[0], image.shape[1])
        
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for (x, y) in keypoints[:, :2]:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(result, (px, py), 4, color, -1)
        return result
    
    def draw_keypoints_with_patches(self, patch_size: int = 32,
                                    keypoint_color: tuple = (0, 255, 0),
                                    patch_color: tuple = (255, 0, 0),
                                    thickness: int = 2) -> np.ndarray:
        """
        Draw keypoints and surrounding patch rectangles on image.
        
        For each keypoint, draws:
        1. A rectangle (patch_size × patch_size) centered on the keypoint
        2. A circle at the keypoint center
        
        Args:
            patch_size: Size of square patch to draw around each keypoint (pixels)
            keypoint_color: RGB color tuple for keypoint circles (default: green)
            patch_color: RGB color tuple for patch rectangles (default: blue)
            thickness: Line thickness for rectangles. -1 to fill.
        
        Returns:
            Image with keypoints and patches drawn
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> result = validator.draw_keypoints_with_patches(patch_size=32)
            >>> cv2.imshow("With patches", result)
        """
        import copy
        result = copy.copy(self.image)
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        half_patch = patch_size // 2
        
        for (x, y) in self.keypoints[:, :2]:
            px = int(x * self.w)
            py = int(y * self.h)
            
            # Draw patch rectangle
            top_left = (max(0, px - half_patch), max(0, py - half_patch))
            bottom_right = (min(self.w, px + half_patch), min(self.h, py + half_patch))
            cv2.rectangle(result, top_left, bottom_right, patch_color, thickness)
            
            # Draw keypoint circle
            cv2.circle(result, (px, py), 4, keypoint_color, -1)
        
        return result
    
    @classmethod
    def load_keypoints(cls, label_path: str, include_visibility: bool = True) -> np.ndarray:
        """
        Load keypoints from a single YOLO pose label file.
        
        Class method for loading keypoints without creating an instance.
        
        Args:
            label_path: Path to .txt label file in YOLO pose format
            include_visibility: If True, returns (N, 3) with visibility.
                              If False, returns (N, 2) only coordinates.
        
        Returns:
            Array of keypoints (N, 2) or (N, 3) depending on include_visibility
            
        Example:
            >>> keypoints = YPImageValidation.load_keypoints("image.txt")
            >>> print(keypoints.shape)  # (9, 3) - 9 keypoints with visibility
        """
        import os
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        
        if not line:
            return np.array([])
        
        # YOLO format: class_id x1 y1 v1 x2 y2 v2 ... xn yn vn
        parts = line.split()
        values = list(map(float, parts[1:]))  # Skip class_id
        
        # Extract keypoints (every 3 values: x, y, visibility)
        keypoints = []
        for i in range(0, len(values), 3):
            if i + 2 < len(values):
                if include_visibility:
                    keypoints.append([values[i], values[i+1], values[i+2]])
                else:
                    keypoints.append([values[i], values[i+1]])
        
        return np.array(keypoints) if keypoints else np.array([])

    
    def visualize(self, patch_size: int = 5):
        """
        Visualize image with keypoints and patches.
        
        Args:
            patch_size: Size of patches to draw
            
        Returns:
            Image with keypoints and patches drawn
        """
        return self.draw_keypoints_with_patches(patch_size=patch_size)

