"""Local Binary Pattern (LBP) feature extraction for keypoint analysis."""

import os
from typing import Union, List
import numpy as np
import cv2


def compute_lbp_value(neighborhood: np.ndarray) -> int:
    """
    Compute LBP value for a SINGLE 3x3 neighborhood.
    
    Low-level pixelwise feature extraction - compares center to 8 neighbors.
    
    Use with YOLOPoseDataset for manual feature extraction:
    
    Example 1 - Process individual neighborhoods:
        >>> from github.utils.validation import YOLOPoseDataset, compute_lbp_value
        >>> import numpy as np
        >>> 
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> sample = dataset[0]
        >>> image = sample['image']
        >>> 
        >>> # Extract 3x3 around a keypoint and compute LBP
        >>> x, y = sample['keypoints'][0]
        >>> x, y = int(x), int(y)
        >>> neighborhood = image[y-1:y+2, x-1:x+2]  # 3x3 patch (must be uint8)
        >>> lbp_value = compute_lbp_value(neighborhood)  # Returns 0-255
    
    Example 2 - Custom processing with dataset:
        >>> def compute_center_lbp(sample):
        ...     image = sample['image']
        ...     h, w = image.shape[:2]
        ...     center_y, center_x = h // 2, w // 2
        ...     neighborhood = image[center_y-1:center_y+2, center_x-1:center_x+2]
        ...     return compute_lbp_value(neighborhood)
        >>> 
        >>> center_lbps = dataset.apply_function_to_all(compute_center_lbp)
    
    Args:
        neighborhood: 3x3 numpy array centered at a pixel
        
    Returns:
        LBP code (0-255) representing the 8-bit binary pattern
    """
    center = neighborhood[1, 1]
    binary_string = ""
    
    # Compare neighbors in clockwise order starting from top-left
    neighbors = [
        neighborhood[0, 0], neighborhood[0, 1], neighborhood[0, 2],
        neighborhood[1, 2], neighborhood[2, 2], neighborhood[2, 1],
        neighborhood[2, 0], neighborhood[1, 0]
    ]
    
    for neighbor in neighbors:
        binary_string += "1" if neighbor >= center else "0"
    
    return int(binary_string, 2)


def compute_lbp_value_variable_radius(image: np.ndarray, y: int, x: int,
                                      radius: int = 1, n_points: int = 8) -> int:
    """
    Compute LBP value using circular sampling at variable radius.
    
    Args:
        image: Grayscale image (H x W)
        y: Y coordinate of center pixel
        x: X coordinate of center pixel
        radius: Radius of circular neighborhood (in pixels)
        n_points: Number of sampling points around the circle
        
    Returns:
        LBP code representing the binary pattern
    """
    h, w = image.shape[:2]
    center_val = image[int(y), int(x)]
    binary_string = ""
    
    # Generate sampling points on a circle
    for i in range(n_points):
        angle = 2.0 * np.pi * i / n_points
        px = int(x + radius * np.cos(angle))
        py = int(y - radius * np.sin(angle))
        
        # Clamp to image boundaries
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        
        neighbor_val = image[py, px]
        binary_string += "1" if neighbor_val >= center_val else "0"
    
    return int(binary_string, 2)


def extract_lbp_patch(image: np.ndarray, x: int, y: int, 
                      patch_size: int = 3) -> Union[int, np.ndarray]:
    """
    Extract and compute LBP value for a single keypoint in an image.
    
    Args:
        image: Grayscale image (H x W)
        x: X coordinate of keypoint (column)
        y: Y coordinate of keypoint (row)
        patch_size: Size of patch around keypoint (default 3x3)
        
    Returns:
        LBP value if patch_size=3, otherwise returns LBP map for larger patches
    """
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    # Clamp coordinates to image boundaries
    x_start = max(0, int(x) - half_size)
    x_end = min(w, int(x) + half_size + 1)
    y_start = max(0, int(y) - half_size)
    y_end = min(h, int(y) + half_size + 1)
    
    # Extract patch
    patch = image[y_start:y_end, x_start:x_end]
    
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        # Pad if patch is too small (near edges)
        patch = np.pad(patch, ((half_size, half_size), (half_size, half_size)), 
                       mode='edge')
    
    if patch_size == 3:
        # For 3x3 patches, compute single LBP value
        neighborhood = patch[y_start:y_end, x_start:x_end]
        if neighborhood.shape != (3, 3):
            return -1  # Invalid patch
        return compute_lbp_value(neighborhood.astype(np.uint8))
    else:
        # For larger patches, compute LBP map
        lbp_map = np.zeros(patch.shape, dtype=np.uint8)
        for py in range(1, patch.shape[0] - 1):
            for px in range(1, patch.shape[1] - 1):
                lbp_map[py, px] = compute_lbp_value(
                    patch[py-1:py+2, px-1:px+2].astype(np.uint8)
                )
        return lbp_map


def compute_lbp_histogram(image: np.ndarray, x: int, y: int, 
                          patch_size: int = 5, bins: int = 256) -> np.ndarray:
    """
    Compute LBP histogram for a SINGLE keypoint using 3x3 neighborhoods.
    
    Use with YOLOPoseDataset for per-keypoint processing:
    
    Example 1 - Process dataset sample:
        >>> from github.utils.validation import YOLOPoseDataset, compute_lbp_histogram
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> 
        >>> sample = dataset[0]
        >>> image = sample['image']
        >>> 
        >>> # Compute histogram for each keypoint individually
        >>> for x, y in sample['keypoints']:
        ...     hist = compute_lbp_histogram(image, int(x), int(y), patch_size=5)
        ...     # hist shape: (256,)
    
    Example 2 - Use with apply_function_to_all:
        >>> def process_per_keypoint(sample, patch_size=5):
        ...     histograms = []
        ...     for x, y in sample['keypoints']:
        ...         h = compute_lbp_histogram(sample['image'], int(x), int(y), patch_size)
        ...         histograms.append(h)
        ...     return np.array(histograms)  # (N, 256)
        >>> 
        >>> all_histograms = dataset.apply_function_to_all(process_per_keypoint, patch_size=5)
    
    Args:
        image: Grayscale image (H x W)
        x: X coordinate of keypoint (column)
        y: Y coordinate of keypoint (row)
        patch_size: Size of patch to analyze around keypoint
        bins: Number of histogram bins (default 256 for LBP)
        
    Returns:
        LBP histogram (256,) - single 256-bin histogram
    """
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    # Extract patch with boundary handling
    x_start = max(0, int(x) - half_size)
    x_end = min(w, int(x) + half_size + 1)
    y_start = max(0, int(y) - half_size)
    y_end = min(h, int(y) + half_size + 1)
    
    patch = image[y_start:y_end, x_start:x_end].astype(np.uint8)
    
    # Pad if necessary
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        patch = np.pad(patch, ((half_size, half_size), (half_size, half_size)), 
                      mode='edge')
    
    # Compute LBP map
    lbp_map = np.zeros(patch.shape, dtype=np.uint8)
    for py in range(1, patch.shape[0] - 1):
        for px in range(1, patch.shape[1] - 1):
            lbp_map[py, px] = compute_lbp_value(patch[py-1:py+2, px-1:px+2])
    
    # Compute histogram
    hist, _ = np.histogram(lbp_map, bins=bins, range=(0, bins))
    return hist.astype(np.float32)


def compute_lbp_histogram_variable_radius(image: np.ndarray, x: int, y: int,
                                         radius: int = 1, n_points: int = 8,
                                         patch_size: int = 5,
                                         bins: int = 256) -> np.ndarray:
    """
    Compute LBP histogram using variable radius circular sampling.
    
    Args:
        image: Grayscale image (H x W)
        x: X coordinate of keypoint center
        y: Y coordinate of keypoint center
        radius: Radius of circular neighborhood (in pixels)
        n_points: Number of sampling points around the circle
        patch_size: Size of patch to analyze
        bins: Number of histogram bins (should match max LBP value)
        
    Returns:
        LBP histogram (1D array)
    """
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    # Extract patch with boundary handling
    x_start = max(0, int(x) - half_size)
    x_end = min(w, int(x) + half_size + 1)
    y_start = max(0, int(y) - half_size)
    y_end = min(h, int(y) + half_size + 1)
    
    patch = image[y_start:y_end, x_start:x_end].astype(np.uint8)
    
    # Pad if necessary
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        patch = np.pad(patch, ((half_size, half_size), (half_size, half_size)), 
                      mode='edge')
    
    # Compute LBP map with variable radius
    lbp_map = np.zeros(patch.shape, dtype=np.uint32)
    py_offset = y_start
    px_offset = x_start
    
    for py in range(patch.shape[0]):
        for px in range(patch.shape[1]):
            img_y = py_offset + py
            img_x = px_offset + px
            if 0 <= img_y < h and 0 <= img_x < w:
                lbp_map[py, px] = compute_lbp_value_variable_radius(
                    image, img_y, img_x, radius, n_points
                )
    
    # Compute histogram
    max_val = 2 ** n_points
    hist, _ = np.histogram(lbp_map, bins=bins, range=(0, min(bins, max_val)))
    return hist.astype(np.float32)


def compute_lbp_for_keypoints(image: Union[np.ndarray, str], 
                              keypoints: np.ndarray,
                              use_histogram: bool = True,
                              patch_size: int = 5) -> np.ndarray:
    """
    Compute LBP patterns for ALL keypoints in an image.
    
    Use with YOLOPoseDataset:
    
    Example - Iterate dataset and compute LBP:
        >>> from github.utils.validation import YOLOPoseDataset, compute_lbp_for_keypoints
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> 
        >>> for sample in dataset:
        ...     hist = compute_lbp_for_keypoints(
        ...         sample['image'],          # numpy array or path
        ...         sample['keypoints'],      # (N, 2) array
        ...         use_histogram=True,
        ...         patch_size=5
        ...     )  # Returns (N, 256)
    
    Args:
        image: Image as numpy array (H x W or H x W x 3) or path to image file
        keypoints: Array of keypoints with shape (N, 2) where N is number of keypoints
                   Each row contains [x, y] coordinates
        use_histogram: If True, return LBP histograms; if False, return LBP values
        patch_size: Size of patch to analyze around each keypoint
        
    Returns:
        Array of LBP values/histograms for each keypoint
        - If use_histogram=False: shape (N,) - one LBP value per keypoint
        - If use_histogram=True: shape (N, 256) - one histogram per keypoint
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    keypoints = np.asarray(keypoints)
    n_keypoints = keypoints.shape[0]
    
    if use_histogram:
        # Return histograms (N x 256)
        lbp_features = np.zeros((n_keypoints, 256), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints):
            lbp_features[i] = compute_lbp_histogram(image, x, y, patch_size)
    else:
        # Return single LBP values (N,)
        lbp_features = np.zeros(n_keypoints, dtype=np.int32)
        for i, (x, y) in enumerate(keypoints):
            lbp_val = extract_lbp_patch(image, x, y, patch_size=3)
            lbp_features[i] = lbp_val if lbp_val >= 0 else 0  # Replace invalid (-1) with 0
    
    return lbp_features


def compute_lbp_for_image(image: Union[np.ndarray, str],
                          keypoints: np.ndarray = None,
                          patch_size: int = 5) -> np.ndarray:
    """
    Compute LBP histogram for ALL keypoints in an image.
    
    Use with YOLOPoseDataset:
    
    Example 1 - Iterate dataset and compute LBP:
        >>> from github.utils.validation import YOLOPoseDataset, compute_lbp_for_image
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> 
        >>> all_histograms = []
        >>> for sample in dataset:
        ...     hist = compute_lbp_for_image(
        ...         sample['image'],          # numpy array or path
        ...         sample['keypoints'],      # (N, 2) array
        ...         patch_size=5
        ...     )
        ...     all_histograms.append(hist)  # shape: (N, 256)
    
    Example 2 - Use with dataset.apply_function_to_all():
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> all_hists = dataset.apply_function_to_all(
        ...     lambda s: compute_lbp_for_image(s['image'], s['keypoints'], 5)
        ... )
    
    Args:
        image: Image array (H x W) or path to image file
        keypoints: Keypoints array (N, 2) with (x, y) coordinates
        patch_size: Size of region around each keypoint
        
    Returns:
        LBP histogram array (N, 256) - one 256-bin histogram per keypoint
    """
    # Handle empty/None keypoints
    if keypoints is None:
        return np.zeros((0, 256), dtype=np.float32)
    
    keypoints = np.asarray(keypoints)
    if keypoints.size == 0 or keypoints.shape[0] == 0:
        return np.zeros((0, 256), dtype=np.float32)
    
    # Use existing function for the computation
    return compute_lbp_for_keypoints(image, keypoints, use_histogram=True, patch_size=patch_size)




def patch_to_binary_decimal(patch: np.ndarray, center_value: float = None) -> dict:
    """
    Convert entire patch to binary matrix and single decimal number.
    
    Compares each pixel in patch to center value: pixel >= center = 1, else 0.
    Flattens binary matrix to binary string, then converts to decimal integer.
    
    Args:
        patch: 2D numpy array of pixel values
        center_value: Center pixel value for comparison. If None, uses patch center pixel.
        
    Returns:
        Dictionary containing:
        - 'binary_matrix': 2D array (0s and 1s) 
        - 'binary_string': Flattened binary as string
        - 'decimal': Integer decimal representation
        - 'patch_shape': Original patch shape (H, W)
        
    Example:
        >>> patch = np.array([[100, 50,  80],
        ...                   [90,  120, 70],
        ...                   [110, 60,  95]])
        >>> result = patch_to_binary_decimal(patch)
        >>> result['decimal']
        16
        >>> result['binary_string']
        '000010000'
        
        For larger patch (7×7):
        >>> patch_7x7 = np.random.randint(0, 256, (7, 7))
        >>> result = patch_to_binary_decimal(patch_7x7)
        >>> result['decimal']  # Single number encoding 49-pixel pattern
        562949953421312
    """
    # Get center value if not provided
    if center_value is None:
        h, w = patch.shape
        center_idx_h, center_idx_w = h // 2, w // 2
        center_value = patch[center_idx_h, center_idx_w]
    
    # Create binary matrix
    binary_matrix = (patch >= center_value).astype(np.uint8)
    
    # Flatten to binary string
    binary_string = ''.join(binary_matrix.flatten().astype(str))
    
    # Convert to decimal
    decimal_value = int(binary_string, 2)
    
    return {
        'binary_matrix': binary_matrix,
        'binary_string': binary_string,
        'decimal': decimal_value,
        'patch_shape': patch.shape
    }


def compute_lbp_binary_decimal(image: Union[np.ndarray, str], 
                               keypoints: np.ndarray,
                                patch_size: int = 5) -> np.ndarray:
    """
    Compute binary decimal representation for ALL keypoints in an image.
    
    Converts each patch to a binary matrix, then to a decimal (0 to 2^(patch_size²)-1).
    
    Use with YOLOPoseDataset:
    
    Example 1 - Iterate dataset and compute binary decimals:
        >>> from github.utils.validation import YOLOPoseDataset, compute_lbp_binary_decimal
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> 
        >>> all_decimals = []
        >>> for sample in dataset:
        ...     decimals = compute_lbp_binary_decimal(
        ...         sample['image'],          # numpy array or path
        ...         sample['keypoints'],      # (N, 2) array
        ...         patch_size=7
        ...     )
        ...     all_decimals.append(decimals)  # shape: (N,) uint64
    
    Example 2 - Use with dataset.apply_function_to_all():
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> all_decs = dataset.apply_function_to_all(
        ...     lambda s: compute_lbp_binary_decimal(s['image'], s['keypoints'], 7)
        ... )
    
    Args:
        image: Image array (H x W) or path to image file
        keypoints: Keypoints array (N, 2) with (x, y) coordinates
        patch_size: Size of region around each keypoint (larger = finer detail)
        
    Returns:
        Binary decimal array (N,) of uint64 values, one per keypoint
    """
    # Load image if path provided
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    keypoints = np.asarray(keypoints)
    if keypoints.size == 0 or keypoints.shape[0] == 0:
        return np.array([], dtype=np.uint64)
    
    h, w = image.shape[:2]
    half_size = patch_size // 2
    decimal_values = []
    
    for x, y in keypoints:
        # Handle normalized coordinates [0, 1]
        if 0 <= x <= 1 and 0 <= y <= 1:
            x_pixel = int(x * w)
            y_pixel = int(y * h)
        else:
            # Assume pixel coordinates
            x_pixel = int(x)
            y_pixel = int(y)
        
        # Extract patch with boundary handling
        x_start = max(0, x_pixel - half_size)
        x_end = min(w, x_pixel + half_size + 1)
        y_start = max(0, y_pixel - half_size)
        y_end = min(h, y_pixel + half_size + 1)
        
        patch = image[y_start:y_end, x_start:x_end].astype(np.uint8)
        
        if patch.size == 0:
            decimal_values.append(0)
            continue
        
        # Pad if necessary to get full patch_size
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # Get binary decimal representation
        result = patch_to_binary_decimal(patch)
        decimal_values.append(result['decimal'])
    
    return np.array(decimal_values, dtype=np.uint64)


class YOLOPoseDataset:
    """
    Class for working with YOLO Pose 1.0 format datasets.
    
    Handles loading, managing, and computing features (LBP histogram, binary decimal)
    for YOLO Pose datasets with keypoint annotations.
    
    Supports multiple input formats:
    - data.yaml (YOLO dataset config) - auto-detects train/val/test split
    - args.yaml (training config) - loads from training directory  
    - train.txt (image list) - direct path to train.txt file
    
    Attributes:
        dataset_file: Path to train.txt file
        dataset_dir: Directory containing images
        labels_dir: Directory containing label files

    USAGE PATTERNS:
    
    Example 1 - Load and iterate (basic):
        >>> dataset = YOLOPoseDataset("data.yaml")  # or "train.txt"
        >>> for sample in dataset:
        ...     image = sample['image']  # numpy array
        ...     keypoints = sample['keypoints']  # (N, 2) array
        ...     image_path = sample['image_path']
    
    Example 2 - Get LBP features (class methods):
        >>> histograms, counts, paths = dataset.get_all_lbp_histograms(patch_size=5)
        >>> decimals, counts, paths = dataset.get_all_lbp_decimals(patch_size=7)
    
    Example 3 - Use standalone functions with dataset:
        >>> from github.utils.validation import compute_lbp_for_image, compute_lbp_binary_decimal
        >>> for sample in dataset:
        ...     hist = compute_lbp_for_image(sample['image'], sample['keypoints'], 5)
        ...     decimals = compute_lbp_binary_decimal(sample['image'], sample['keypoints'], 7)
    
    Example 4 - Apply custom function to all samples:
        >>> def custom_feature(sample, patch_size=5):
        ...     from github.utils.validation import compute_lbp_for_image
        ...     return compute_lbp_for_image(sample['image'], sample['keypoints'], patch_size)
        >>> results = dataset.apply_function_to_all(custom_feature, patch_size=7)
    """
    
    def __init__(self, dataset_file: str, labels_dir: str = None):
        """
        Initialize YOLO Pose dataset.
        
        Args:
            dataset_file: Path to data.yaml, args.yaml, or train.txt file
                - data.yaml: YOLO dataset config (auto-detects train/val/test split)
                - args.yaml: YOLO training config (loads from training directory)
                - train.txt: File with relative image paths, one per line
            labels_dir: Directory with label files. If None, infers from dataset_file parent
        """
        dataset_file_abs = os.path.abspath(dataset_file)
        
        # Check if input is data.yaml
        if dataset_file.endswith('data.yaml') or dataset_file.endswith('data.yml'):
            self.dataset_file, self.dataset_dir = self._load_from_data_yaml(dataset_file_abs, labels_dir)
        else:
            self.dataset_file = dataset_file_abs
            self.dataset_dir = os.path.dirname(self.dataset_file)
        
        # Infer labels directory
        if labels_dir is None:
            self.labels_dir = os.path.join(self.dataset_dir, "labels")
        else:
            self.labels_dir = os.path.abspath(labels_dir)
        
        # Load image paths from train.txt
        with open(self.dataset_file, "r") as f:
            self.image_rel_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        # Filter to valid image/label pairs
        self._valid_pairs = self._validate_pairs()
    
    @staticmethod
    def _load_from_data_yaml(data_yaml_path: str, labels_dir: str = None) -> tuple:
        """
        Parse data.yaml and find train.txt, val.txt, or test.txt automatically.
        
        Args:
            data_yaml_path: Absolute path to data.yaml
            labels_dir: Optional explicit labels directory
            
        Returns:
            (dataset_file_path, dataset_dir) tuple
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for data.yaml support. Install: pip install pyyaml")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get dataset root path from data.yaml
        dataset_root = data_config.get('path', os.path.dirname(data_yaml_path))
        if not os.path.isabs(dataset_root):
            dataset_root = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), dataset_root))
        
        # Try to find train.txt, val.txt, test.txt in priority order
        split_priorities = ['train', 'val', 'test']
        found_file = None
        found_dir = None
        
        for split in split_priorities:
            split_path = data_config.get(split)
            if not split_path:
                # Try default directory names
                split_path = os.path.join(dataset_root, split)
            else:
                # Make absolute if relative
                if not os.path.isabs(split_path):
                    split_path = os.path.join(dataset_root, split_path)
            
            txt_file = os.path.join(split_path, f'{split}.txt')
            if os.path.exists(txt_file):
                found_file = txt_file
                found_dir = split_path
                print(f"Loading dataset from: {found_file}")
                break
        
        if not found_file:
            raise FileNotFoundError(
                f"Could not find train.txt, val.txt, or test.txt in {dataset_root}\n"
                f"Checked paths based on data.yaml: {data_config}"
            )
        
        return found_file, found_dir
    
    def _validate_pairs(self):
        """Check which image/label pairs exist and are valid."""
        from . import helpers
        
        valid = []
        for rel_path in self.image_rel_paths:
            image_path = os.path.join(self.dataset_dir, rel_path)
            if not os.path.exists(image_path):
                continue
            
            label_name = os.path.splitext(os.path.basename(rel_path))[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)
            if not os.path.exists(label_path):
                continue
            
            # Try loading to ensure validity
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                keypoints = helpers.load_keypoints(label_path)
                if keypoints is None or keypoints.size == 0:
                    continue
                valid.append((image_path, label_path))
            except Exception:
                continue
        
        return valid
    
    @property
    def stats(self) -> dict:
        """Get dataset statistics."""
        total_images = len(self._valid_pairs)
        
        # Count total keypoints
        total_keypoints = 0
        for _, label_path in self._valid_pairs:
            try:
                from . import helpers
                kpts = helpers.load_keypoints(label_path)
                if kpts is not None:
                    total_keypoints += len(kpts)
            except Exception:
                pass
        
        return {
            'total_images': total_images,
            'total_keypoints': total_keypoints,
            'avg_keypoints_per_image': total_keypoints / total_images if total_images > 0 else 0,
            'dataset_file': self.dataset_file,
            'labels_dir': self.labels_dir
        }
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self._valid_pairs)
    
    def __iter__(self):
        """Iterate through dataset, yielding samples WITHOUT LBP features.
        
        Use iterate_with_lbp(patch_size) if you need LBP features.
        """
        from . import helpers
        
        for image_path, label_path in self._valid_pairs:
            try:
                # Load image and keypoints
                image = cv2.imread(image_path)
                keypoints = helpers.load_keypoints(label_path)
                
                if image is None or keypoints is None or keypoints.size == 0:
                    continue
                
                yield {
                    'image': image,
                    'keypoints': keypoints,
                    'image_path': image_path,
                    'label_path': label_path
                }
            except Exception:
                continue
    
    def iterate_with_lbp(self, patch_size: int = 5):
        """Iterate through dataset, yielding samples WITH LBP features.
        
        Args:
            patch_size: Size of patch for LBP computation
            
        Yields:
            Dict with keys: image, keypoints, image_path, label_path, lbp_histogram, lbp_decimal
        """
        from . import helpers
        
        for image_path, label_path in self._valid_pairs:
            try:
                # Load image and keypoints
                image = cv2.imread(image_path)
                keypoints = helpers.load_keypoints(label_path)
                
                if image is None or keypoints is None or keypoints.size == 0:
                    continue
                
                # Compute LBP features
                lbp_histogram = compute_lbp_for_image(image, keypoints, patch_size)
                lbp_decimal = compute_lbp_binary_decimal(image, keypoints, patch_size)
                
                yield {
                    'image': image,
                    'keypoints': keypoints,
                    'image_path': image_path,
                    'label_path': label_path,
                    'lbp_histogram': lbp_histogram,
                    'lbp_decimal': lbp_decimal
                }
            except Exception:
                continue
    
    def __getitem__(self, idx: int) -> dict:
        """Get sample by index without computing features."""
        from . import helpers
        
        if idx >= len(self._valid_pairs):
            raise IndexError(f"Sample index {idx} out of range {len(self._valid_pairs)}")
        
        image_path, label_path = self._valid_pairs[idx]
        
        image = cv2.imread(image_path)
        keypoints = helpers.load_keypoints(label_path)
        
        return {
            'image': image,
            'keypoints': keypoints,
            'image_path': image_path,
            'label_path': label_path
        }
    
    def get_sample_with_features(self, idx: int, patch_size: int = 5) -> dict:
        """Get sample by index WITH LBP features computed.
        
        Args:
            idx: Sample index
            patch_size: Size of patch for LBP computation
        """
        sample = self[idx]
        
        sample['lbp_histogram'] = compute_lbp_for_image(
            sample['image'], sample['keypoints'], patch_size
        )
        sample['lbp_decimal'] = compute_lbp_binary_decimal(
            sample['image'], sample['keypoints'], patch_size
        )
        
        return sample
    
    def get_all_lbp_histograms(self, patch_size: int = 5, verbose: bool = True) -> tuple:
        """
        Compute and return LBP histograms for all samples.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            
        Returns:
            Tuple of (lbp_features, keypoints_per_image, image_paths):
            - lbp_features: Array (total_keypoints, 256)
            - keypoints_per_image: List of keypoint counts per image
            - image_paths: List of image paths
        """
        lbp_list = []
        paths = []
        counts = []
        
        iterator = self.iterate_with_lbp(patch_size)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self), desc=f"Computing LBP histograms (patch_size={patch_size})")
            except ImportError:
                pass
        
        for sample in iterator:
            lbp_list.append(sample['lbp_histogram'])
            paths.append(sample['image_path'])
            counts.append(len(sample['keypoints']))
        
        all_lbp = np.vstack(lbp_list) if lbp_list else np.zeros((0, 256), dtype=np.float32)
        return all_lbp, counts, paths
    
    def get_all_lbp_decimals(self, patch_size: int = 5, verbose: bool = True) -> tuple:
        """
        Compute and return binary decimal representations for all samples.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            
        Returns:
            Tuple of (decimals, keypoints_per_image, image_paths):
            - decimals: Array (total_keypoints,) of uint64
            - keypoints_per_image: List of keypoint counts per image
            - image_paths: List of image paths
        """
        decimal_list = []
        paths = []
        counts = []
        
        iterator = self.iterate_with_lbp(patch_size)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self), desc=f"Computing binary decimals (patch_size={patch_size})")
            except ImportError:
                pass
        
        for sample in iterator:
            decimal_list.append(sample['lbp_decimal'])
            paths.append(sample['image_path'])
            counts.append(len(sample['keypoints']))
        
        all_decimals = np.concatenate(decimal_list) if decimal_list else np.array([], dtype=np.uint64)
        return all_decimals, counts, paths
    
    def _tqdm_iter(self, desc: str):
        """Helper for progress bar during iteration."""
        try:
            from tqdm import tqdm
            return tqdm(self, total=len(self), desc=desc)
        except ImportError:
            return self
    
    def create_dataframe(self, patch_size: int = 5) -> 'pd.DataFrame':
        """
        Create pandas DataFrame with all samples and features.
        
        Args:
            patch_size: Size of patch for LBP computation
        
        Returns:
            DataFrame with columns:
            - image_path: Path to image
            - label_path: Path to label
            - keypoint_idx: Index of keypoint
            - lbp_histogram: LBP histogram (256 values)
            - lbp_decimal: LBP decimal (uint64)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for create_dataframe()")
        
        data = []
        for sample in self.iterate_with_lbp(patch_size):
            for i, (hist, dec) in enumerate(zip(sample['lbp_histogram'], 
                                                  sample['lbp_decimal'])):
                data.append({
                    'image_path': sample['image_path'],
                    'label_path': sample['label_path'],
                    'keypoint_idx': i,
                    'lbp_histogram': list(hist),
                    'lbp_decimal': dec
                })
        
        df = pd.DataFrame(data)
        return df
    
    def visualize_sample(self, idx: int = 0, patch_size: int = 5):
        """
        Visualize a sample with keypoints and patches.
        
        Args:
            idx: Sample index to visualize
            patch_size: Size of patches to draw
        """
        from . import helpers
        
        sample = self[idx]
        result = helpers.draw_keypoints_with_patches(
            sample['image'], 
            sample['keypoints'],
            patch_size=patch_size
        )
        
        return result
    
    def apply_function_to_all(self, func, *args, **kwargs):
        """
        Apply a standalone function to all samples for custom processing.
        
        Enables using standalone functions from this module with dataset samples.
        
        Args:
            func: Callable that takes (sample_dict) as first argument
            *args, **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results, one per sample
            
        Example - use standalone compute_lbp_for_image with dataset:
            >>> from github.utils.validation import compute_lbp_for_image
            >>> dataset = YOLOPoseDataset('train.txt')
            >>> 
            >>> def extract_lbp(sample, patch_size=5):
            ...     return compute_lbp_for_image(
            ...         sample['image'],
            ...         sample['keypoints'],
            ...         patch_size=patch_size
            ...     )
            >>> 
            >>> all_histograms = dataset.apply_function_to_all(extract_lbp, patch_size=7)
            >>> # all_histograms is a list with one (N, 256) histogram per sample
        """
        results = []
        for sample in self:
            result = func(sample, *args, **kwargs)
            results.append(result)
        return results


class YOLOPoseDataset:
    """
    Class for working with YOLO Pose 1.0 format datasets.
    
    Handles loading, managing, and computing features (LBP histogram, binary decimal)
    for YOLO Pose datasets with keypoint annotations.
    
    Supports multiple input formats:
    - data.yaml (YOLO dataset config) - auto-detects train/val/test split
    - args.yaml (training config) - loads from training directory  
    - train.txt (image list) - direct path to train.txt file
    
    Attributes:
        dataset_file: Path to train.txt file
        dataset_dir: Directory containing images
        labels_dir: Directory containing label files

    USAGE PATTERNS:
    
    Example 1 - Load and iterate (basic):
        >>> dataset = YOLOPoseDataset("data.yaml")  # or "train.txt"
        >>> for sample in dataset:
        ...     image = sample['image']  # numpy array
        ...     keypoints = sample['keypoints']  # (N, 2) array
        ...     image_path = sample['image_path']
    
    Example 2 - Get LBP features (class methods):
        >>> histograms, counts, paths = dataset.get_all_lbp_histograms(patch_size=5)
        >>> decimals, counts, paths = dataset.get_all_lbp_decimals(patch_size=7)
    
    Example 3 - Use standalone functions with dataset:
        >>> from github.utils.validation import compute_lbp_for_image, compute_lbp_binary_decimal
        >>> for sample in dataset:
        ...     hist = compute_lbp_for_image(sample['image'], sample['keypoints'], 5)
        ...     decimals = compute_lbp_binary_decimal(sample['image'], sample['keypoints'], 7)
    
    Example 4 - Apply custom function to all samples:
        >>> def custom_feature(sample, patch_size=5):
        ...     from github.utils.validation import compute_lbp_for_image
        ...     return compute_lbp_for_image(sample['image'], sample['keypoints'], patch_size)
        >>> results = dataset.apply_function_to_all(custom_feature, patch_size=7)
    """
    
    def __init__(self, dataset_file: str, labels_dir: str = None):
        """
        Initialize YOLO Pose dataset.
        
        Args:
            dataset_file: Path to data.yaml, args.yaml, or train.txt file
                - data.yaml: YOLO dataset config (auto-detects train/val/test split)
                - args.yaml: YOLO training config (loads from training directory)
                - train.txt: File with relative image paths, one per line
            labels_dir: Directory with label files. If None, infers from dataset_file parent
        """
        dataset_file_abs = os.path.abspath(dataset_file)
        
        # Check if input is data.yaml
        if dataset_file.endswith('data.yaml') or dataset_file.endswith('data.yml'):
            self.dataset_file, self.dataset_dir = self._load_from_data_yaml(dataset_file_abs, labels_dir)
        else:
            self.dataset_file = dataset_file_abs
            self.dataset_dir = os.path.dirname(self.dataset_file)
        
        # Infer labels directory
        if labels_dir is None:
            self.labels_dir = os.path.join(self.dataset_dir, "labels")
        else:
            self.labels_dir = os.path.abspath(labels_dir)
        
        # Load image paths from train.txt
        with open(self.dataset_file, "r") as f:
            self.image_rel_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        # Filter to valid image/label pairs
        self._valid_pairs = self._validate_pairs()
    
    @staticmethod
    def _load_from_data_yaml(data_yaml_path: str, labels_dir: str = None) -> tuple:
        """
        Parse data.yaml and find train.txt, val.txt, or test.txt automatically.
        
        Args:
            data_yaml_path: Absolute path to data.yaml
            labels_dir: Optional explicit labels directory
            
        Returns:
            (dataset_file_path, dataset_dir) tuple
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for data.yaml support. Install: pip install pyyaml")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get dataset root path from data.yaml
        dataset_root = data_config.get('path', os.path.dirname(data_yaml_path))
        if not os.path.isabs(dataset_root):
            dataset_root = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), dataset_root))
        
        # Try to find train.txt, val.txt, test.txt in priority order
        split_priorities = ['train', 'val', 'test']
        found_file = None
        found_dir = None
        
        for split in split_priorities:
            split_path = data_config.get(split)
            if not split_path:
                # Try default directory names
                split_path = os.path.join(dataset_root, split)
            else:
                # Make absolute if relative
                if not os.path.isabs(split_path):
                    split_path = os.path.join(dataset_root, split_path)
            
            txt_file = os.path.join(split_path, f'{split}.txt')
            if os.path.exists(txt_file):
                found_file = txt_file
                found_dir = split_path
                print(f"Loading dataset from: {found_file}")
                break
        
        if not found_file:
            raise FileNotFoundError(
                f"Could not find train.txt, val.txt, or test.txt in {dataset_root}\n"
                f"Checked paths based on data.yaml: {data_config}"
            )
        
        return found_file, found_dir
    
    def _validate_pairs(self):
        """Check which image/label pairs exist and are valid."""
        from . import helpers
        
        valid = []
        for rel_path in self.image_rel_paths:
            image_path = os.path.join(self.dataset_dir, rel_path)
            if not os.path.exists(image_path):
                continue
            
            label_name = os.path.splitext(os.path.basename(rel_path))[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)
            if not os.path.exists(label_path):
                continue
            
            # Try loading to ensure validity
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                keypoints = helpers.load_keypoints(label_path)
                if keypoints is None or keypoints.size == 0:
                    continue
                valid.append((image_path, label_path))
            except Exception:
                continue
        
        return valid
    
    @property
    def stats(self) -> dict:
        """Get dataset statistics."""
        total_images = len(self._valid_pairs)
        
        # Count total keypoints
        total_keypoints = 0
        for _, label_path in self._valid_pairs:
            try:
                from . import helpers
                kpts = helpers.load_keypoints(label_path)
                if kpts is not None:
                    total_keypoints += len(kpts)
            except Exception:
                pass
        
        return {
            'total_images': total_images,
            'total_keypoints': total_keypoints,
            'avg_keypoints_per_image': total_keypoints / total_images if total_images > 0 else 0,
            'dataset_file': self.dataset_file,
            'labels_dir': self.labels_dir
        }
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self._valid_pairs)
    
    def __iter__(self):
        """Iterate through dataset, yielding samples WITHOUT LBP features.
        
        Use iterate_with_lbp(patch_size) if you need LBP features.
        """
        from . import helpers
        
        for image_path, label_path in self._valid_pairs:
            try:
                # Load image and keypoints
                image = cv2.imread(image_path)
                keypoints = helpers.load_keypoints(label_path)
                
                if image is None or keypoints is None or keypoints.size == 0:
                    continue
                
                yield {
                    'image': image,
                    'keypoints': keypoints,
                    'image_path': image_path,
                    'label_path': label_path
                }
            except Exception:
                continue
    
    def iterate_with_lbp(self, patch_size: int = 5):
        """Iterate through dataset, yielding samples WITH LBP features.
        
        Args:
            patch_size: Size of patch for LBP computation
            
        Yields:
            Dict with keys: image, keypoints, image_path, label_path, lbp_histogram, lbp_decimal
        """
        from . import helpers
        
        for image_path, label_path in self._valid_pairs:
            try:
                # Load image and keypoints
                image = cv2.imread(image_path)
                keypoints = helpers.load_keypoints(label_path)
                
                if image is None or keypoints is None or keypoints.size == 0:
                    continue
                
                # Compute LBP features
                lbp_histogram = compute_lbp_for_image(image, keypoints, patch_size)
                lbp_decimal = compute_lbp_binary_decimal(image, keypoints, patch_size)
                
                yield {
                    'image': image,
                    'keypoints': keypoints,
                    'image_path': image_path,
                    'label_path': label_path,
                    'lbp_histogram': lbp_histogram,
                    'lbp_decimal': lbp_decimal
                }
            except Exception:
                continue
    
    def __getitem__(self, idx: int) -> dict:
        """Get sample by index without computing features."""
        from . import helpers
        
        if idx >= len(self._valid_pairs):
            raise IndexError(f"Sample index {idx} out of range {len(self._valid_pairs)}")
        
        image_path, label_path = self._valid_pairs[idx]
        
        image = cv2.imread(image_path)
        keypoints = helpers.load_keypoints(label_path)
        
        return {
            'image': image,
            'keypoints': keypoints,
            'image_path': image_path,
            'label_path': label_path
        }
    
    def get_sample_with_features(self, idx: int, patch_size: int = 5) -> dict:
        """Get sample by index WITH LBP features computed.
        
        Args:
            idx: Sample index
            patch_size: Size of patch for LBP computation
        """
        sample = self[idx]
        
        sample['lbp_histogram'] = compute_lbp_for_image(
            sample['image'], sample['keypoints'], patch_size
        )
        sample['lbp_decimal'] = compute_lbp_binary_decimal(
            sample['image'], sample['keypoints'], patch_size
        )
        
        return sample
    
    def get_all_lbp_histograms(self, patch_size: int = 5, verbose: bool = True) -> tuple:
        """
        Compute and return LBP histograms for all samples.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            
        Returns:
            Tuple of (lbp_features, keypoints_per_image, image_paths):
            - lbp_features: Array (total_keypoints, 256)
            - keypoints_per_image: List of keypoint counts per image
            - image_paths: List of image paths
        """
        lbp_list = []
        paths = []
        counts = []
        
        iterator = self.iterate_with_lbp(patch_size)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self), desc=f"Computing LBP histograms (patch_size={patch_size})")
            except ImportError:
                pass
        
        for sample in iterator:
            lbp_list.append(sample['lbp_histogram'])
            paths.append(sample['image_path'])
            counts.append(len(sample['keypoints']))
        
        all_lbp = np.vstack(lbp_list) if lbp_list else np.zeros((0, 256), dtype=np.float32)
        return all_lbp, counts, paths
    
    def get_all_lbp_decimals(self, patch_size: int = 5, verbose: bool = True) -> tuple:
        """
        Compute and return binary decimal representations for all samples.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            
        Returns:
            Tuple of (decimals, keypoints_per_image, image_paths):
            - decimals: Array (total_keypoints,) of uint64
            - keypoints_per_image: List of keypoint counts per image
            - image_paths: List of image paths
        """
        decimal_list = []
        paths = []
        counts = []
        
        iterator = self.iterate_with_lbp(patch_size)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self), desc=f"Computing binary decimals (patch_size={patch_size})")
            except ImportError:
                pass
        
        for sample in iterator:
            decimal_list.append(sample['lbp_decimal'])
            paths.append(sample['image_path'])
            counts.append(len(sample['keypoints']))
        
        all_decimals = np.concatenate(decimal_list) if decimal_list else np.array([], dtype=np.uint64)
        return all_decimals, counts, paths
    
    def _tqdm_iter(self, desc: str):
        """Helper for progress bar during iteration."""
        try:
            from tqdm import tqdm
            return tqdm(self, total=len(self), desc=desc)
        except ImportError:
            return self
    
    def create_dataframe(self, patch_size: int = 5) -> 'pd.DataFrame':
        """
        Create pandas DataFrame with all samples and features.
        
        Args:
            patch_size: Size of patch for LBP computation
        
        Returns:
            DataFrame with columns:
            - image_path: Path to image
            - label_path: Path to label
            - keypoint_idx: Index of keypoint
            - lbp_histogram: LBP histogram (256 values)
            - lbp_decimal: LBP decimal (uint64)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for create_dataframe()")
        
        data = []
        for sample in self.iterate_with_lbp(patch_size):
            for i, (hist, dec) in enumerate(zip(sample['lbp_histogram'], 
                                                  sample['lbp_decimal'])):
                data.append({
                    'image_path': sample['image_path'],
                    'label_path': sample['label_path'],
                    'keypoint_idx': i,
                    'lbp_histogram': list(hist),
                    'lbp_decimal': dec
                })
        
        df = pd.DataFrame(data)
        return df
    
    def visualize_sample(self, idx: int = 0, patch_size: int = 5):
        """
        Visualize a sample with keypoints and patches.
        
        Args:
            idx: Sample index to visualize
            patch_size: Size of patches to draw
        """
        from . import helpers
        
        sample = self[idx]
        result = helpers.draw_keypoints_with_patches(
            sample['image'], 
            sample['keypoints'],
            patch_size=patch_size
        )
        
        return result
    
    def apply_function_to_all(self, func, *args, **kwargs):
        """
        Apply a standalone function to all samples for custom processing.
        
        Enables using standalone functions from this module with dataset samples.
        
        Args:
            func: Callable that takes (sample_dict) as first argument
            *args, **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results, one per sample
            
        Example - use standalone compute_lbp_for_image with dataset:
            >>> from github.utils.validation import compute_lbp_for_image
            >>> dataset = YOLOPoseDataset('train.txt')
            >>> 
            >>> def extract_lbp(sample, patch_size=5):
            ...     return compute_lbp_for_image(
            ...         sample['image'],
            ...         sample['keypoints'],
            ...         patch_size=patch_size
            ...     )
            >>> 
            >>> all_histograms = dataset.apply_function_to_all(extract_lbp, patch_size=7)
            >>> # all_histograms is a list with one (N, 256) histogram per sample
        """
        results = []
        for sample in self:
            result = func(sample, *args, **kwargs)
            results.append(result)
        return results
