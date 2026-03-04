"""Dataset-level validation methods: LBP features and duplicate detection."""

import os
from typing import List, Tuple, Optional, Union, Callable, Dict, Any, Sequence
import numpy as np
import cv2
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip
import faiss
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from .image_validation import YPImageValidation
from ..config import (
    CLIP_MODEL_NAME,
    CLIP_MODEL_PRETRAINED,
    VALID_IMAGE_EXTENSIONS,
    DEFAULT_K_NEIGHBORS,
)


# Mapping of string keys to sklearn regressor constructors
DEFAULT_REGRESSORS = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
    'bayesian_ridge': BayesianRidge,
    'huber': HuberRegressor,
    'ransac': RANSACRegressor,
    'svr': SVR,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'knn': KNeighborsRegressor,
    'mlp': MLPRegressor,
}


class YPSetValidation:
    """
    Validation for entire dataset - computes features across all images.
    
    Wraps YOLOPoseDataset and YPImageValidation to compute LBP features
    and validate sequential distances across the entire dataset.
    
    Use this class for dataset-level operations like batch feature extraction
    and distance anomaly detection.
    Use YPImageValidation for single image operations.
    
    USAGE PATTERNS:
    
    Example 1 - Get all LBP histograms:
        >>> from github.utils.validation import YPSetValidation, YOLOPoseDataset
        >>> dataset = YOLOPoseDataset('train.txt')
        >>> validator = YPSetValidation(dataset)
        >>> histograms, counts_per_image, paths = validator.get_all_lbp_histograms(patch_size=5)
    
    Example 2 - Check where sequential distances are anomalous:
        >>> validator = YPSetValidation(dataset)
        >>> models = validator.train_distance_models(model_factory="linear")
        >>> anomalies = validator.predict_distance_anomalies(models, threshold_percentile=95)
        >>> # Returns list of images ranked by distance prediction error
    
    Example 3 - Get dataset statistics:
        >>> stats = validator.get_stats()
    """
    
    def __init__(self, dataset):
        """
        Initialize dataset validator.
        
        Args:
            dataset: YOLOPoseDataset instance
        """
        self.dataset = dataset
    
    def get_all_lbp_histograms(self, patch_size: int = 5, verbose: bool = True, 
                               visibility_threshold: float = 0.5) -> tuple:
        """
        Compute LBP histograms for all visible keypoints in entire dataset.
        
        Skips hidden/occluded keypoints based on visibility threshold.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Tuple of (lbp_features, keypoints_per_image, image_paths):
            - lbp_features: Array (total_visible_keypoints, 256)
            - keypoints_per_image: List of visible keypoint counts per image
            - image_paths: List of image paths
        """
        lbp_list = []
        paths = []
        counts = []
        
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), 
                              desc=f"Computing LBP histograms (patch_size={patch_size}, visibility_threshold={visibility_threshold})")
            except ImportError:
                pass
        
        for sample in iterator:
            # Compute LBP for this image
            validator = YPImageValidation(sample['image'], sample['keypoints'])
            histograms = validator.compute_lbp_histograms(patch_size, visibility_threshold)
            
            lbp_list.append(histograms)
            paths.append(sample['image_path'])
            counts.append(len(histograms))  # Count of visible keypoints
        
        all_lbp = np.vstack(lbp_list) if lbp_list else np.zeros((0, 256), dtype=np.float32)
        return all_lbp, counts, paths
    
    def get_all_lbp_decimals(self, patch_size: int = 5, verbose: bool = True,
                             visibility_threshold: float = 0.5) -> tuple:
        """
        Compute binary decimal representations for all visible keypoints in entire dataset.
        
        Skips hidden/occluded keypoints based on visibility threshold.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Tuple of (decimals, keypoints_per_image, image_paths):
            - decimals: Array (total_visible_keypoints,) of uint64
            - keypoints_per_image: List of visible keypoint counts per image
            - image_paths: List of image paths
        """
        decimal_list = []
        paths = []
        counts = []
        
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), 
                              desc=f"Computing binary decimals (patch_size={patch_size}, visibility_threshold={visibility_threshold})")
            except ImportError:
                pass
        
        for sample in iterator:
            # Compute decimals for this image
            validator = YPImageValidation(sample['image'], sample['keypoints'])
            decimals = validator.compute_lbp_decimals(patch_size, visibility_threshold)
            
            decimal_list.append(decimals)
            paths.append(sample['image_path'])
            counts.append(len(decimals))  # Count of visible keypoints
        
        all_decimals = np.concatenate(decimal_list) if decimal_list else np.array([], dtype=np.uint64)
        return all_decimals, counts, paths
    
    def create_dataframe(self, patch_size: int = 5, visibility_threshold: float = 0.5) -> 'pd.DataFrame':
        """
        Create pandas DataFrame with all visible samples and features from entire dataset.
        
        Skips hidden/occluded keypoints based on visibility threshold.
        
        Args:
            patch_size: Size of patch for LBP computation
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
        
        Returns:
            DataFrame with columns:
            - image_path: Path to image
            - label_path: Path to label
            - keypoint_idx: Index of (visible) keypoint
            - lbp_histogram: LBP histogram (256 values)
            - lbp_decimal: LBP decimal (uint64)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for create_dataframe()")
        
        data = []
        for sample in self.dataset:
            # Compute features for this image
            validator = YPImageValidation(sample['image'], sample['keypoints'])
            histograms = validator.compute_lbp_histograms(patch_size, visibility_threshold)
            decimals = validator.compute_lbp_decimals(patch_size, visibility_threshold)
            
            for i, (hist, dec) in enumerate(zip(histograms, decimals)):
                data.append({
                    'image_path': sample['image_path'],
                    'label_path': sample['label_path'],
                    'keypoint_idx': i,
                    'lbp_histogram': list(hist),
                    'lbp_decimal': dec
                })
        
        df = pd.DataFrame(data)
        return df
    
    def get_stats(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with:
            - total_images: Number of images
            - total_keypoints: Total keypoints across all images
            - avg_keypoints_per_image: Average keypoints per image
            - dataset_file: Path to dataset file
            - labels_dir: Path to labels directory
        """
        return self.dataset.stats
    
    # ==================== Keypoint Loading (Batch) ====================
    
    @staticmethod
    def load_all_keypoints(dataset_file: str, labels_dir: str = None, split: str = None) -> List[np.ndarray]:
        """
        Load all keypoints from a YOLO pose dataset in one operation.
        
        Static method for batch keypoint loading from either data.yaml or train.txt.
        
        Args:
            dataset_file: Path to either data.yaml or train.txt file
            labels_dir: Path to labels directory. If None, inferred from dataset_file parent.
            split: For data.yaml, which split to load ('train', 'val', 'test').
                  If None, loads 'train' by default.
        
        Returns:
            List of keypoint arrays, one per object across all files.
            Each array has shape (N_keypoints, 2) with normalized [x, y] coordinates.
            
        Example:
            >>> # From data.yaml
            >>> kpts = YPSetValidation.load_all_keypoints("data.yaml", split="train")
            >>> 
            >>> # From train.txt
            >>> kpts = YPSetValidation.load_all_keypoints("train.txt")
        """
        from . import helpers
        return helpers.load_all_keypoints(dataset_file, labels_dir=labels_dir, split=split)
    
    # ==================== Drawing & Visualization ====================
    
    def draw_all_keypoints(self, output_folder: str = "output_keypoints",
                          color: tuple = (0, 255, 0), verbose: bool = True) -> List[str]:
        """
        Draw keypoints on all images in dataset and save results.
        
        Args:
            output_folder: Folder to save images with drawn keypoints
            color: RGB color tuple (R, G, B) in range [0, 255]
            verbose: Show progress bar
            
        Returns:
            List of output image paths
            
        Example:
            >>> validator = YPSetValidation(dataset)
            >>> output_paths = validator.draw_all_keypoints("output", color=(0, 255, 0))
        """
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        output_paths = []
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), desc="Drawing keypoints")
            except ImportError:
                pass
        
        for sample in iterator:
            validator = YPImageValidation(sample['image'], sample['keypoints'])
            result = validator.draw_keypoints(color=color)
            
            # Save result
            filename = os.path.basename(sample['image_path'])
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            output_paths.append(output_path)
        
        return output_paths
    
    def draw_all_keypoints_with_patches(self, output_folder: str = "output_patches",
                                       patch_size: int = 32,
                                       keypoint_color: tuple = (0, 255, 0),
                                       patch_color: tuple = (255, 0, 0),
                                       verbose: bool = True) -> List[str]:
        """
        Draw keypoints and patches on all images in dataset and save results.
        
        Args:
            output_folder: Folder to save images with patches
            patch_size: Size of square patches to draw
            keypoint_color: RGB color tuple for keypoint circles
            patch_color: RGB color tuple for patch rectangles
            verbose: Show progress bar
            
        Returns:
            List of output image paths
            
        Example:
            >>> validator = YPSetValidation(dataset)
            >>> output_paths = validator.draw_all_keypoints_with_patches(
            ...     "output_patches", patch_size=32
            ... )
        """
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        output_paths = []
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), desc="Drawing key points with patches")
            except ImportError:
                pass
        
        for sample in iterator:
            validator = YPImageValidation(sample['image'], sample['keypoints'])
            result = validator.draw_keypoints_with_patches(
                patch_size=patch_size,
                keypoint_color=keypoint_color,
                patch_color=patch_color
            )
            
            # Save result
            filename = os.path.basename(sample['image_path'])
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            output_paths.append(output_path)
        
        return output_paths
    
    def get_distance_matrices(self, verbose: bool = True, visibility_threshold: float = 0.5) -> Tuple[np.ndarray, List[str]]:
        """Extract sequential distance matrices from all keypoint sequences in dataset.
        
        Computes sequential distances between consecutive visible keypoints in each image.
        
        Args:
            verbose: Show progress bar
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Tuple of (distance_matrix, image_paths):
            - distance_matrix: Array of shape (num_images, num_keypoints-1) with sequential distances
            - image_paths: List of image paths corresponding to each row
        """
        distance_list = []
        paths = []
        
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), 
                              desc=f"Extracting sequential distances (visibility_threshold={visibility_threshold})")
            except ImportError:
                pass
        
        for sample in iterator:
            image = sample['image']
            kpts = sample['keypoints']
            if len(kpts) < 2:
                continue
            
            # Use YPImageValidation instance method
            validator = YPImageValidation(image, kpts)
            distances = validator.sequential_distances(visibility_threshold=visibility_threshold)
            
            if len(distances) > 0:
                distance_list.append(distances)
                paths.append(sample['image_path'])
        
        distance_matrix = np.vstack(distance_list) if distance_list else np.zeros((0, 0), dtype=np.float32)
        return distance_matrix, paths
    
    def train_distance_models(
        self,
        model_factory: Union[Callable[[], BaseEstimator], str] = "linear",
        exclude_endpoints: bool = False,
        verbose: bool = True,
    ) -> List[BaseEstimator]:
        """Train regression models to predict one distance from the others.
        
        Extracts sequential distances from dataset and trains a separate regressor
        for each distance index. Each model predicts that distance from all others.
        
        Args:
            model_factory: Either a callable returning a fresh estimator or a string
                key: linear, ridge, lasso, elasticnet, bayesian_ridge, huber, ransac,
                svr, decision_tree, random_forest, gradient_boosting, knn, mlp
            exclude_endpoints: If True, ignore first and last distances (for interior
                distances only). Defaults to False.
            verbose: Show progress bar
            
        Returns:
            List of fitted sklearn estimators
        """
        if BaseEstimator is None:
            raise ImportError("scikit-learn is required for training distance models")

        # Get distance matrix from dataset
        distance_matrix, _ = self.get_distance_matrices(verbose=verbose)
        
        if distance_matrix.size == 0:
            raise ValueError("Dataset has no valid keypoint sequences")

        # resolve string names to constructors
        if isinstance(model_factory, str):
            key = model_factory.lower()
            if key not in DEFAULT_REGRESSORS:
                raise ValueError(f"unknown regressor '{model_factory}'; valid options are: {', '.join(sorted(DEFAULT_REGRESSORS.keys()))}")
            model_factory = DEFAULT_REGRESSORS[key]

        distance_matrix = np.asarray(distance_matrix)
        if distance_matrix.ndim != 2:
            raise ValueError(f"distance_matrix must be 2‑D, got {distance_matrix.ndim}‑D")

        num_samples, num_distances = distance_matrix.shape
        if num_samples < 2:
            raise ValueError("At least two samples are required to train models")

        models: List[BaseEstimator] = []
        indices = list(range(num_distances))
        if exclude_endpoints:
            if num_distances < 3:
                return []
            indices = list(range(1, num_distances - 1))

        for idx in indices:
            X = np.delete(distance_matrix, idx, axis=1)
            y = distance_matrix[:, idx]
            model = model_factory()
            model.fit(X, y)
            models.append(model)

        return models
    
    def predict_distance_anomalies(
        self,
        models: Sequence[BaseEstimator],
        exclude_endpoints: bool = False,
        threshold_percentile: float = 95.0,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Identify images with anomalous sequential distances using trained models.
        
        Uses trained models to predict distances and compares with actual values.
        Images where predictions deviate significantly are flagged as anomalies.
        
        Args:
            models: List of fitted estimators from train_distance_models()
            exclude_endpoints: Must match the flag used in train_distance_models()
            threshold_percentile: Percentile of error to flag as anomaly (0-100)
            verbose: Show progress bar
            
        Returns:
            List of dicts with:
            - image_path: Path to image
            - error: Mean absolute error for this image
            - is_anomaly: Boolean, True if error exceeds threshold
            - rank: Ranking by error (1 = worst)
        """
        if BaseEstimator is None:
            raise ImportError("scikit-learn is required")

        distance_matrix, paths = self.get_distance_matrices(verbose=verbose)
        
        if distance_matrix.size == 0:
            raise ValueError("Dataset has no valid keypoint sequences")

        num_distances = distance_matrix.shape[1]
        indices = list(range(num_distances))
        if exclude_endpoints:
            indices = list(range(1, num_distances - 1))

        if len(models) != len(indices):
            raise ValueError("number of models does not match expected number of distances")

        # Compute predictions
        preds = []
        for model, idx in zip(models, indices):
            X = np.delete(distance_matrix, idx, axis=1)
            preds.append(model.predict(X))
        
        predictions = np.column_stack(preds)
        
        # For each image, compute error
        actual = distance_matrix[:, indices]
        errors = np.mean(np.abs(predictions - actual), axis=1)
        
        # Determine anomaly threshold
        threshold = np.percentile(errors, threshold_percentile)
        
        # Sort by error
        sorted_idx = np.argsort(-errors)
        
        report = []
        for rank, idx in enumerate(sorted_idx, start=1):
            report.append({
                'rank': rank,
                'image_path': paths[idx],
                'error': float(errors[idx]),
                'is_anomaly': errors[idx] > threshold,
            })
        
        return report
    
    # ==================== Duplicate & Similarity Detection (CLIP-based) ====================
    
    @staticmethod
    def _load_clip_model(device: str) -> Tuple:
        """Load CLIP model for image embedding.
        
        Args:
            device: Device to load model on ("cuda" or "cpu")
            
        Returns:
            Tuple of (model, preprocess) for CLIP
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_MODEL_PRETRAINED
        )
        model = model.to(device)
        model.eval()
        return model, preprocess
    
    @staticmethod
    def _compute_embeddings(
        image_folder: str,
        model,
        preprocess,
        device: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute CLIP embeddings for all images in folder.
        
        Args:
            image_folder: Path to folder with images
            model: CLIP model instance
            preprocess: CLIP preprocessing function
            device: Device for inference
            
        Returns:
            Tuple of (embeddings array, list of filenames)
        """
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        embeddings = []
        filenames = []

        with torch.no_grad():
            for filename in tqdm(image_files, desc="Computing embeddings"):
                path = os.path.join(image_folder, filename)

                try:
                    image = Image.open(path).convert("RGB")
                    image = preprocess(image).unsqueeze(0).to(device)

                    embedding = model.encode_image(image)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                    embeddings.append(embedding.cpu().numpy())
                    filenames.append(filename)
                except Exception:
                    continue

        embeddings = np.vstack(embeddings).astype("float32")
        return embeddings, filenames
    
    @staticmethod
    def _build_similarity_groups(
        embeddings: np.ndarray,
        threshold: float,
        k: int = DEFAULT_K_NEIGHBORS
    ) -> List[List[int]]:
        """Build groups of similar images using FAISS.
        
        Args:
            embeddings: Array of image embeddings
            threshold: Similarity threshold for grouping
            k: Number of neighbors to search
            
        Returns:
            List of groups, where each group is a list of image indices
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        similarities, indices = index.search(embeddings, k)

        graph = defaultdict(set)

        for i in range(len(embeddings)):
            for j in range(1, k):
                if similarities[i][j] > threshold:
                    neighbor = indices[i][j]
                    graph[i].add(neighbor)
                    graph[neighbor].add(i)

        visited = set()
        groups = []

        for node in range(len(embeddings)):
            if node not in visited:
                stack = [node]
                component = []

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current])

                if len(component) > 1:
                    groups.append(component)

        return groups
    
    @staticmethod
    def generate_embeddings(
        image_folder: str,
        device: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate CLIP embeddings for images in folder.
        
        Args:
            image_folder: Path to folder with images
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
            
        Returns:
            Tuple of (embeddings array, list of filenames)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = YPSetValidation._load_clip_model(device)

        embeddings, filenames = YPSetValidation._compute_embeddings(
            image_folder,
            model,
            preprocess,
            device
        )

        return embeddings, filenames
    
    @staticmethod
    def find_near_duplicates(
        embeddings: np.ndarray,
        filenames: List[str],
        k: int = DEFAULT_K_NEIGHBORS,
        threshold: float = 0.99
    ) -> List[Tuple[str, str, float]]:
        """Find near-duplicate images based on embeddings.
        
        Args:
            embeddings: Array of image embeddings
            filenames: List of image filenames
            k: Number of neighbors to search
            threshold: Similarity threshold
            
        Returns:
            List of (file1, file2, similarity) tuples for duplicates
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        similarities, indices = index.search(embeddings, k)

        duplicates = []

        for i in range(len(filenames)):
            for j in range(1, k):
                if similarities[i][j] > threshold:
                    duplicates.append(
                        (filenames[i], filenames[indices[i][j]], float(similarities[i][j]))
                    )

        return duplicates
    
    @staticmethod
    def build_similarity_groups(
        embeddings: np.ndarray,
        threshold: float = 0.95,
        k: int = DEFAULT_K_NEIGHBORS
    ) -> List[List[int]]:
        """Build groups of similar images using FAISS.
        
        Args:
            embeddings: Array of image embeddings
            threshold: Similarity threshold for grouping
            k: Number of neighbors to search
            
        Returns:
            List of groups, where each group is a list of image indices
        """
        return YPSetValidation._build_similarity_groups(embeddings, threshold, k)
