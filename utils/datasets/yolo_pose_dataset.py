"""YOLO Pose dataset loading and management."""

import os
from typing import Union, Dict, Any
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class YOLOSample:
    """Sample from YOLOPoseDataset with image and keypoints."""
    image: np.ndarray
    keypoints: np.ndarray
    image_path: str
    label_path: str
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)


class YOLOPoseImage:
    """
    Represents a single image in YOLO Pose format with keypoint annotations.
    
    Handles loading and managing a single image with its corresponding label file.
    Provides a clean interface for per-image operations and feature extraction.
    
    For feature extraction (LBP, visibility checking, distances), use:
    - YPImageValidation: Wraps YOLOPoseImage with LBP and distance methods
    
    Attributes:
        image_path: Absolute path to image file
        label_path: Absolute path to label file (.txt)
        image: Loaded image as numpy array (H x W x 3 or H x W)
        keypoints: Keypoint array (N, 2) with normalized [x, y] OR (N, 3) with [x, y, visibility]
        image_name: Filename of the image
        label_name: Filename of the label file
    
    Example 1 - Direct usage:
        >>> img = YOLOPoseImage("path/to/image.jpg", "path/to/image.txt")
        >>> print(f"Keypoints: {img.keypoints.shape}")  # (N, 2) or (N, 3)
        >>> cv2.imshow("Image", img.image)
    
    Example 2 - With feature extraction:
        >>> from utils.validation import YPImageValidation
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> validator = YPImageValidation(yolo_img.image, yolo_img.keypoints)
        >>> hist = validator.compute_lbp_histograms(patch_size=5)
        >>> distances = validator.sequential_distances(visibility_threshold=0.5)
    
    Example 3 - Get as dict (compatible with YOLOPoseDataset):
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> sample = yolo_img.to_dict()
        >>> image = sample['image']
        >>> keypoints = sample['keypoints']
    """
    
    def __init__(self, image_path: Union[str, np.ndarray], label_path: str = None, 
                 include_visibility: bool = True):
        """
        Initialize YOLOPoseImage.
        
        Args:
            image_path: Path to image file or numpy array
            label_path: Path to YOLO label file (.txt). Required if image_path is path.
                       If None and image_path is str, infers label_path from image path.
            include_visibility: If True, load visibility scores as 3rd dimension.
                              If False, load only [x, y] coordinates.
        
        Raises:
            FileNotFoundError: If image or label file doesn't exist
            ValueError: If image cannot be loaded
        """
        # Load image
        if isinstance(image_path, str):
            self.image_path = os.path.abspath(image_path)
            if not os.path.exists(self.image_path):
                raise FileNotFoundError(f"Image not found: {self.image_path}")
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            self.image_name = os.path.basename(self.image_path)
        else:
            # Image provided as numpy array
            self.image = image_path
            self.image_path = None
            self.image_name = None
        
        # Load keypoints
        if label_path is None:
            if self.image_path is None:
                raise ValueError("label_path required when image_path is numpy array")
            # Infer label path from image path
            img_name = os.path.splitext(self.image_name)[0]
            img_dir = os.path.dirname(self.image_path)
            label_path = os.path.join(img_dir, '..', 'labels', img_name + '.txt')
        
        self.label_path = os.path.abspath(label_path)
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        
        self.label_name = os.path.basename(self.label_path)
        
        # Load keypoints from label file
        self._load_keypoints(include_visibility)
    
    @classmethod
    def from_yaml_index(cls, data_yaml: str, index: int, split: str = 'train',
                       include_visibility: bool = True) -> 'YOLOPoseImage':
        """
        Load image by index from a data.yaml config.
        
        Args:
            data_yaml: Path to data.yaml file
            index: Index of image in the split
            split: Which split to load ('train', 'val', 'test'). Default 'train'.
            include_visibility: If True, load visibility scores as 3rd dimension.
            
        Returns:
            YOLOPoseImage instance
            
        Raises:
            FileNotFoundError: If data.yaml or dataset files not found
            IndexError: If index is out of range
            
        Example:
            >>> img = YOLOPoseImage.from_dataset_index("data.yaml", index=0, split="train")
            >>> distances = img.get_sequential_distances()
        """
        dataset = YOLOPoseDataset(data_yaml)
        sample = dataset[index]
        
        return cls(
            image_path=sample['image_path'],
            label_path=sample['label_path'],
            include_visibility=include_visibility
        )
    
    @classmethod
    def from_txt_index(cls, file_list_path: str, index: int, labels_dir: str = None,
                      include_visibility: bool = True) -> 'YOLOPoseImage':
        """
        Load image by index from a file list (train.txt, val.txt, etc.).
        
        Args:
            file_list_path: Path to file list (.txt) with image paths, one per line
            index: Index of image in the file list
            labels_dir: Path to labels directory. If None, inferred from file_list_path parent.
                       Defaults to: os.path.dirname(file_list_path)/../labels/
            include_visibility: If True, load visibility scores as 3rd dimension.
            
        Returns:
            YOLOPoseImage instance
            
        Raises:
            FileNotFoundError: If file list or dataset files not found
            IndexError: If index is out of range
            
        Example:
            >>> img = YOLOPoseImage.from_file_index("train.txt", index=42)
            >>> distances = img.get_sequential_distances()
            >>> 
            >>> # From val.txt with custom labels directory
            >>> img = YOLOPoseImage.from_file_index("val.txt", 0, labels_dir="custom/labels")
        """
        dataset = YOLOPoseDataset(file_list_path, labels_dir=labels_dir)
        sample = dataset[index]
        
        return cls(
            image_path=sample['image_path'],
            label_path=sample['label_path'],
            include_visibility=include_visibility
        )
    
    def _load_keypoints(self, include_visibility: bool = True):
        """Load keypoints from label file."""
        from ..validation.helpers import load_keypoints
        self.keypoints = load_keypoints(self.label_path, include_visibility=include_visibility)
    
    @property
    def num_keypoints(self) -> int:
        """Get number of keypoints."""
        return len(self.keypoints)
    
    @property
    def image_size(self) -> tuple:
        """Get image dimensions as (height, width)."""
        return self.image.shape[:2]
    
    @property
    def has_visibility(self) -> bool:
        """Check if keypoints include visibility scores."""
        return self.keypoints.shape[1] >= 3
    
    def get_sequential_distances(self, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Get sequential distances between consecutive visible keypoints.
        
        Computes Euclidean distances from keypoint[i] to keypoint[i+1] for 
        consecutive visible pairs. Skips distances involving hidden/occluded 
        keypoints (visibility < threshold).
        
        Args:
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
        
        Returns:
            Array of distances between consecutive visible keypoints.
            Returns empty array if fewer than 2 visible keypoints.
            
        Example:
            >>> yolo_img = YOLOPoseImage("image.jpg", "label.txt")
            >>> distances = yolo_img.get_sequential_distances(visibility_threshold=0.5)
            >>> print(distances)  # [0.045, 0.052, 0.031, ...]
            >>> print(f"Mean distance: {distances.mean():.4f}")
        """
        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(self.image, self.keypoints)
        return validator.sequential_distances(visibility_threshold=visibility_threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with YOLOPoseDataset.
        
        Returns:
            Dictionary with keys: 'image', 'keypoints', 'image_path', 'label_path'
        """
        return {
            'image': self.image,
            'keypoints': self.keypoints,
            'image_path': self.image_path,
            'label_path': self.label_path
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"YOLOPoseImage(image={self.image_name}, "
                f"keypoints={self.num_keypoints}, "
                f"visibility={self.has_visibility})")


class YOLOPoseDataset:
    """
    Class for working with YOLO Pose 1.0 format datasets.
    
    Handles loading, managing, and iterating through YOLO Pose datasets with keypoint annotations.
    
    For feature extraction, use:
    - YPImageValidation: For single image operations
    - YPSetValidation: For dataset-level operations (LBP histograms, decimals, dataframes)
    
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
    
    Example 2 - Get LBP features (use YPSetValidation):
        >>> from github.utils.validation import YPSetValidation
        >>> validator = YPSetValidation(dataset)
        >>> histograms, counts, paths = validator.get_all_lbp_histograms(patch_size=5)
        >>> decimals, counts, paths = validator.get_all_lbp_decimals(patch_size=7)
    
    Example 3 - Single image features (use YPImageValidation):
        >>> from github.utils.validation import YPImageValidation
        >>> sample = dataset[0]
        >>> validator = YPImageValidation(sample['image'], sample['keypoints'])
        >>> hist = validator.compute_lbp_histograms(patch_size=5)
    
    Example 4 - Use YOLOPoseImage for per-image operations:
        >>> from utils.datasets import YOLOPoseImage
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> print(f"Keypoints: {yolo_img.keypoints.shape}")
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
        from ..validation.image_validation import YPImageValidation
        
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
                keypoints = YPImageValidation.load_keypoints(label_path)
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
                from ..validation.image_validation import YPImageValidation
                kpts = YPImageValidation.load_keypoints(label_path)
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
        from ..validation.image_validation import YPImageValidation
        
        for image_path, label_path in self._valid_pairs:
            try:
                # Load image and keypoints
                image = cv2.imread(image_path)
                keypoints = YPImageValidation.load_keypoints(label_path)
                
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
    
    def __getitem__(self, idx: int) -> YOLOSample:
        """Get sample by index without computing features.
        
        Returns a YOLOSample object with attributes: image, keypoints, image_path, label_path
        
        Also supports dict-like access for backward compatibility:
        - sample.image or sample['image']
        - sample.keypoints or sample['keypoints']
        - sample.image_path or sample['image_path']
        - sample.label_path or sample['label_path']
        """
        from ..validation.image_validation import YPImageValidation
        
        if idx >= len(self._valid_pairs):
            raise IndexError(f"Sample index {idx} out of range {len(self._valid_pairs)}")
        
        image_path, label_path = self._valid_pairs[idx]
        
        image = cv2.imread(image_path)
        keypoints = YPImageValidation.load_keypoints(label_path)
        
        return YOLOSample(
            image=image,
            keypoints=keypoints,
            image_path=image_path,
            label_path=label_path
        )    
    def get_sequential_distances(self, idx: int, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Get sequential distances for a single sample by index.
        
        Computes Euclidean distances between consecutive visible keypoints.
        Skips distances involving hidden/occluded keypoints (visibility < threshold).
        
        Args:
            idx: Sample index in dataset
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
        
        Returns:
            Array of distances between consecutive visible keypoints.
            Returns empty array if fewer than 2 visible keypoints.
            
        Raises:
            IndexError: If idx is out of range
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> distances = dataset.get_sequential_distances(0, visibility_threshold=0.5)
            >>> print(distances)  # [0.045, 0.052, 0.031, ...]
            >>> print(f"Mean distance: {distances.mean():.4f}")
        """
        sample = self[idx]
        image = sample['image']
        keypoints = sample['keypoints']
        
        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(image, keypoints)
        return validator.sequential_distances(visibility_threshold=visibility_threshold)