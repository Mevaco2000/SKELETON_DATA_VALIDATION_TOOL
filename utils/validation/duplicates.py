"""Duplicate detection using CLIP embeddings and FAISS similarity search."""

import os
from typing import Tuple, List, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import faiss
from collections import defaultdict

from ..config import (
    CLIP_MODEL_NAME,
    CLIP_MODEL_PRETRAINED,
    VALID_IMAGE_EXTENSIONS,
    DEFAULT_K_NEIGHBORS,
)


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


def generate_clip_embeddings(
    image_folder: str,
    device: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """Generate CLIP embeddings for images.
    
    Args:
        image_folder: Path to folder with images
        device: Device to use ("cuda" or "cpu"). Auto-detects if None.
        
    Returns:
        Tuple of (embeddings array, list of filenames)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = _load_clip_model(device)

    embeddings, filenames = _compute_embeddings(
        image_folder,
        model,
        preprocess,
        device
    )

    return embeddings, filenames


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
