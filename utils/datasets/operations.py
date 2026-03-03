"""Dataset manipulation operations (flatten, merge, split, convert)."""

import os
import shutil
import random
import zipfile
from typing import Optional
from tqdm import tqdm

from ..config import DEFAULT_VAL_RATIO, DEFAULT_RANDOM_SEED


def flatten_cvat_yolo_pose(
    input_root: str,
    output_root: str,
    copy_data_yaml: bool = True
) -> None:
    """Flatten nested CVAT YOLO pose dataset structure.
    
    Converts nested directory structure into flat images/ and labels/ folders.
    
    Args:
        input_root: Root directory of nested dataset
        output_root: Root directory for flattened output
        copy_data_yaml: Whether to copy data.yaml file
    """
    os.makedirs(output_root, exist_ok=True)

    out_images = os.path.join(output_root, "images")
    out_labels = os.path.join(output_root, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    label_index = {}

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".txt"):
                name = os.path.splitext(file)[0]
                label_index[name] = os.path.join(root, file)

    train_lines = []
    existing_names = set()

    image_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append((root, file))

    for root, file in tqdm(image_files, desc="Flattening dataset"):

        img_src = os.path.join(root, file)
        name, ext = os.path.splitext(file)

        new_name = file
        counter = 1

        while new_name in existing_names:
            new_name = f"{name}_{counter}{ext}"
            counter += 1

        existing_names.add(new_name)

        shutil.copy(img_src, os.path.join(out_images, new_name))

        if name in label_index:
            shutil.copy(
                label_index[name],
                os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
            )

        train_lines.append(f"images/{new_name}\n")

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(train_lines)

    if copy_data_yaml:
        for root, _, files in os.walk(input_root):
            if "data.yaml" in files:
                shutil.copy(
                    os.path.join(root, "data.yaml"),
                    os.path.join(output_root, "data.yaml")
                )
                break


def merge_yolo_pose_datasets(
    dataset1_root: str,
    dataset2_root: str,
    output_root: str
) -> None:
    """Merge two YOLO pose datasets.
    
    Combines images and labels from two datasets, handling filename conflicts.
    
    Args:
        dataset1_root: Root directory of first dataset
        dataset2_root: Root directory of second dataset
        output_root: Root directory for merged output
    """
    os.makedirs(output_root, exist_ok=True)
    out_images = os.path.join(output_root, "images")
    out_labels = os.path.join(output_root, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    merged_train_lines = []
    existing_names = set()

    def process_dataset(dataset_root: str, prefix: str) -> None:
        nonlocal merged_train_lines, existing_names

        images_path = os.path.join(dataset_root, "images")
        labels_path = os.path.join(dataset_root, "labels")
        train_txt_path = os.path.join(dataset_root, "train.txt")

        with open(train_txt_path, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Merging {prefix}", leave=False):
            img_rel_path = line.strip()
            img_name = os.path.basename(img_rel_path)

            name, ext = os.path.splitext(img_name)

            if img_name in existing_names:
                new_name = f"{prefix}_{img_name}"
            else:
                new_name = img_name

            existing_names.add(new_name)

            shutil.copy(
                os.path.join(images_path, img_name),
                os.path.join(out_images, new_name)
            )

            label_name = name + ".txt"
            src_label = os.path.join(labels_path, label_name)

            if os.path.exists(src_label):
                shutil.copy(
                    src_label,
                    os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
                )

            merged_train_lines.append(f"images/{new_name}\n")

    process_dataset(dataset1_root, "ds1")
    process_dataset(dataset2_root, "ds2")

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(merged_train_lines)

    data_yaml_1 = os.path.join(dataset1_root, "data.yaml")
    if os.path.exists(data_yaml_1):
        shutil.copy(data_yaml_1, os.path.join(output_root, "data.yaml"))


def split_yolo_pose_dataset(
    dataset_root: str,
    output_root: str,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
    dataset_path: Optional[str] = None
) -> None:
    """Split YOLO pose dataset into train and validation sets.
    
    Args:
        dataset_root: Root directory of input dataset
        output_root: Root directory for split output
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        dataset_path: Base path for data.yaml (defaults to output_root)
    """
    random.seed(seed)

    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")

    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)

    out_images_train = os.path.join(output_root, "images", "train")
    out_images_val = os.path.join(output_root, "images", "val")
    out_labels_train = os.path.join(output_root, "labels", "train")
    out_labels_val = os.path.join(output_root, "labels", "val")

    os.makedirs(out_images_train, exist_ok=True)
    os.makedirs(out_images_val, exist_ok=True)
    os.makedirs(out_labels_train, exist_ok=True)
    os.makedirs(out_labels_val, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - val_ratio))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def copy_files(file_list, img_out_dir: str, lbl_out_dir: str):
        txt_lines = []
        split_name = os.path.basename(img_out_dir)

        for img_name in tqdm(file_list, desc=f"Copying {split_name}", leave=False):
            name, ext = os.path.splitext(img_name)

            shutil.copy(
                os.path.join(images_root, img_name),
                os.path.join(img_out_dir, img_name)
            )

            label_path = os.path.join(labels_root, name + ".txt")
            if os.path.exists(label_path):
                shutil.copy(
                    label_path,
                    os.path.join(lbl_out_dir, name + ".txt")
                )

            txt_lines.append(f"images/{split_name}/{img_name}\n")

        return txt_lines

    train_lines = copy_files(train_files, out_images_train, out_labels_train)
    val_lines = copy_files(val_files, out_images_val, out_labels_val)

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(train_lines)

    with open(os.path.join(output_root, "val.txt"), "w") as f:
        f.writelines(val_lines)

    data_yaml_src = os.path.join(dataset_root, "data.yaml")
    data_yaml_dst = os.path.join(output_root, "data.yaml")

    yaml_content = ""

    if os.path.exists(data_yaml_src):
        with open(data_yaml_src, "r") as f:
            yaml_content = f.read()

    lines = yaml_content.splitlines()
    cleaned_lines = [
        line for line in lines
        if not line.strip().startswith(("path:", "train:", "val:"))
    ]

    if dataset_path is None:
        dataset_path = output_root

    cleaned_lines.insert(0, f"path: {dataset_path}")
    cleaned_lines.insert(1, "train: images/train")
    cleaned_lines.insert(2, "val: images/val")

    with open(data_yaml_dst, "w") as f:
        f.write("\n".join(cleaned_lines) + "\n")


def convert_yolo_pose_to_cvat(
    dataset_root: str,
    output_zip_dir: str
) -> None:
    """Convert YOLO pose dataset to CVAT format with ZIP archives.
    
    Args:
        dataset_root: Root directory of YOLO dataset
        output_zip_dir: Directory for output ZIP files
    """
    os.makedirs(output_zip_dir, exist_ok=True)

    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")
    train_txt_path = os.path.join(dataset_root, "train.txt")
    data_yaml_path = os.path.join(dataset_root, "data.yaml")

    nested_images_dir = os.path.join(labels_root, "train", "images")
    os.makedirs(nested_images_dir, exist_ok=True)

    # Copy images to nested structure
    image_files = [
        f for f in os.listdir(images_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in tqdm(image_files, desc="Copying images to nested structure"):
        shutil.copy(
            os.path.join(images_root, img_name),
            os.path.join(nested_images_dir, img_name)
        )

    # Update train.txt
    if not os.path.exists(train_txt_path):
        raise FileNotFoundError("train.txt not found")

    with open(train_txt_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        filename = os.path.basename(line.strip())
        updated_lines.append(f"data/images/train/{filename}\n")

    with open(train_txt_path, "w") as f:
        f.writelines(updated_lines)

    # ZIP images
    images_zip_path = os.path.join(output_zip_dir, "images.zip")

    with zipfile.ZipFile(images_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(images_root):
            for file in tqdm(files, desc="Zipping images", leave=False):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_root)
                zipf.write(full_path, rel_path)

    # ZIP annotations
    annotations_zip_path = os.path.join(output_zip_dir, "annotations.zip")

    with zipfile.ZipFile(annotations_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Labels
        for root, _, files in os.walk(labels_root):
            for file in tqdm(files, desc="Zipping labels", leave=False):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_root)
                zipf.write(full_path, rel_path)

        # data.yaml
        if os.path.exists(data_yaml_path):
            zipf.write(
                data_yaml_path,
                os.path.relpath(data_yaml_path, dataset_root)
            )

        # train.txt
        zipf.write(
            train_txt_path,
            os.path.relpath(train_txt_path, dataset_root)
        )


def extract_image_subset(
    images_path: str,
    output_dir: str,
    a: int,
    b: int
) -> None:
    """Extract images by index range from a folder.
    
    Copies images with indices [a, b] based on sorted folder listing.
    Useful for extracting specific subsets from large image collections.
    
    Args:
        images_path: Path to source folder with images
        output_dir: Path to destination folder
        a: Start index (inclusive)
        b: End index (inclusive)
        
    Returns:
        None. Prints number of images copied.
        
    Raises:
        IndexError: If index range is out of bounds
        ValueError: If a > b
        
    Example:
        >>> extract_image_subset(
        ...     images_path="./images",
        ...     output_dir="./subset",
        ...     a=0,
        ...     b=100
        ... )
        ✔ Skopiowano 101 obrazów do ./subset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of images
    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    n = len(image_files)

    if a < 0 or b >= n:
        raise IndexError(f"Index range must be in [0–{n-1}]")
    if a > b:
        raise ValueError("a must be <= b")

    selected = image_files[a:b+1]

    for img_file in selected:
        shutil.copy(
            os.path.join(images_path, img_file),
            os.path.join(output_dir, img_file)
        )

    print(f"✔ Skopiowano {len(selected)} obrazów do {output_dir}")
