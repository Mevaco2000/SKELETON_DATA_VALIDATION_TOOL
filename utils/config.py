"""Configuration and constants for the utils package."""

import os

# Get absolute path to models directory
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_for_analysis")

MODEL_REGISTRY = {
    "custom": None,  # path will be provided dynamically
}

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VALID_LABEL_EXTENSION = ".txt"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_MODEL_PRETRAINED = "openai"

DEFAULT_THRESHOLD_SIMILARITY = 0.98
DEFAULT_THRESHOLD_DUPLICATES = 0.99
DEFAULT_K_NEIGHBORS = 10

DEFAULT_YOLO_EPOCHS = 100
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_BATCH = 16
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42
