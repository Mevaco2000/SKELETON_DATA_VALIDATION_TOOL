"""Analysis of image groups and visualization."""

import os
import shutil
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from openpyxl import Workbook

from .helpers import load_keypoints, sequential_distances, draw_keypoints


def generate_group_visualizations(
    groups: List[List[int]],
    filenames: List[str],
    image_folder: str,
    label_folder: str,
    output_folder: str
) -> None:
    """Generate visualizations for image groups.
    
    Args:
        groups: List of groups, each group is a list of image indices
        filenames: List of image filenames
        image_folder: Path to folder with images
        label_folder: Path to folder with labels
        output_folder: Path to save visualizations
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    for i, group in enumerate(tqdm(groups, desc="Generating visualizations")):

        base_idx = group[0]
        base_img = cv2.imread(os.path.join(image_folder, filenames[base_idx]))
        image = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        legend_entries = []

        for idx in group:
            img_name = filenames[idx]
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_folder, label_name)

            if not os.path.exists(label_path):
                continue

            kp = load_keypoints(label_path)
            color = tuple(random.randint(0, 255) for _ in range(3))

            image = draw_keypoints(image, kp, color)
            legend_entries.append((f"(idx={idx})", color))

        output_path = os.path.join(output_folder, f"group_{i}.png")

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")

        for name, color in legend_entries:
            plt.plot([], [], marker='o',
                     color=np.array(color)/255,
                     linestyle='None',
                     label=name)

        plt.legend(fontsize=6, loc="upper right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def save_groups_analysis(
    groups: List[List[int]],
    filenames: List[str],
    label_folder: str,
    output_json: str
) -> None:
    """Save group analysis to JSON file.
    
    Args:
        groups: List of groups
        filenames: List of image filenames
        label_folder: Path to folder with labels
        output_json: Path to save JSON analysis
    """
    data = {}

    for i, group in enumerate(tqdm(groups, desc="Saving group analysis")):
        group_key = f"group_{i}"
        data[group_key] = []

        for idx in sorted(group):

            img_name = filenames[idx]
            label_path = os.path.join(
                label_folder,
                os.path.splitext(img_name)[0] + ".txt"
            )

            entry = {"filename": img_name, "index": int(idx)}

            if os.path.exists(label_path):
                kp = load_keypoints(label_path)
                dist_vector = sequential_distances(kp)
                entry["distance_vector"] = [
                    round(float(d), 2) for d in dist_vector
                ]

            data[group_key].append(entry)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def analyze_hidden_keypoints(label_folder: str, output_json: str) -> None:
    """Analyze hidden keypoints in labels.
    
    Args:
        label_folder: Path to folder with labels
        output_json: Path to save analysis
    """
    results = []

    label_files = [
        f for f in os.listdir(label_folder)
        if f.endswith(".txt")
    ]

    for filename in tqdm(label_files, desc="Analyzing hidden keypoints"):

        path = os.path.join(label_folder, filename)

        with open(path, "r") as f:
            lines = f.readlines()

        hidden_keypoints = []

        for object_id, line in enumerate(lines):
            parts = line.strip().split()
            keypoints_data = list(map(float, parts[5:]))

            for i in range(0, len(keypoints_data), 3):
                if int(keypoints_data[i + 2]) == 1:
                    hidden_keypoints.append({
                        "object_id": object_id,
                        "keypoint_index": i // 3
                    })

        if hidden_keypoints:
            results.append({
                "filename": filename,
                "hidden_keypoints": hidden_keypoints
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def export_groups_analysis_to_excel(json_path: str, output_path: str) -> None:
    """Export group analysis to Excel with statistics.
    
    For each group:
    - Calculates mean for each distance (d1, d2, ...)
    - Calculates global group mean
    - Sorts groups in descending order by global mean
    - Saves to Excel
    
    Args:
        json_path: Path to groups_analysis.json
        output_path: Path for output Excel file
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_groups = []

    for group_name, group_items in data.items():

        vectors = [
            item.get("distance_vector", [])
            for item in group_items
            if "distance_vector" in item
        ]

        if not vectors:
            continue

        max_len = max(len(v) for v in vectors)

        # Pad vectors to same length
        padded = []
        for v in vectors:
            padded_vec = v + [np.nan] * (max_len - len(v))
            padded.append(padded_vec)

        padded = np.array(padded, dtype=float)

        # Mean for each keypoint distance
        mean_per_distance = np.nanmean(padded, axis=0)

        # Global group mean
        global_mean = float(np.nanmean(padded))

        processed_groups.append({
            "group_name": group_name,
            "items": group_items,
            "mean_per_distance": mean_per_distance,
            "global_mean": global_mean
        })

    # Sort descending by global mean
    processed_groups.sort(
        key=lambda x: x["global_mean"],
        reverse=True
    )

    # Export to Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Groups_sorted_by_mean"

    current_row = 1

    for group in tqdm(processed_groups, desc="Exporting sorted groups"):

        ws.cell(row=current_row, column=1,
                value=f"{group['group_name']} (mean={round(group['global_mean'], 4)})")
        current_row += 1

        ws.cell(row=current_row, column=1, value="index")

        max_vector_len = len(group["mean_per_distance"])

        for i in range(max_vector_len):
            ws.cell(row=current_row, column=2 + i, value=f"d{i+1}")

        current_row += 1

        # Data rows
        for item in group["items"]:

            ws.cell(row=current_row, column=1, value=item.get("index"))

            vector = item.get("distance_vector", [])
            for i in range(len(vector)):
                ws.cell(row=current_row, column=2 + i, value=vector[i])

            current_row += 1

        # Mean row
        ws.cell(row=current_row, column=1, value="MEAN_PER_DISTANCE")
        for i, val in enumerate(group["mean_per_distance"]):
            ws.cell(row=current_row, column=2 + i,
                    value=float(round(val, 4)))

        current_row += 2

    wb.save(output_path)


# ---------------------------------------------------------------------------
# distance prediction helpers using scikit-learn
# ---------------------------------------------------------------------------
from typing import Callable, Sequence, List, Union

try:
    from sklearn.base import BaseEstimator
    # a handful of common regressors; more can be added later
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        BayesianRidge,
        HuberRegressor,
        RANSACRegressor,
    )
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    # registry mapping short names to constructors
    DEFAULT_REGRESSORS = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
        "bayesian_ridge": BayesianRidge,
        "huber": HuberRegressor,
        "ransac": RANSACRegressor,
        "svr": SVR,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "knn": KNeighborsRegressor,
        "mlp": MLPRegressor,
    }
except ImportError:
    # scikit-learn is an optional dependency; functions will raise if used without it
    BaseEstimator = None  # type: ignore
    DEFAULT_REGRESSORS = {}


def distance_matrix_from_keypoints(
    keypoints_list: Sequence[np.ndarray]
) -> np.ndarray:
    """Convert a sequence of keypoint arrays into a distance matrix.

    Each element of ``keypoints_list`` should be an ``(N,2)`` array.  The
    returned matrix has shape ``(num_samples, N-1)`` where each row contains the
    sequential distances for that sample.  This is just a convenience wrapper
    around :func:`~utils.validation.helpers.sequential_distances`.
    """
    from .helpers import sequential_distances

    return np.vstack([sequential_distances(kp) for kp in keypoints_list])


def train_distance_models(
    distance_matrix: np.ndarray,
    model_factory: Union[Callable[[], BaseEstimator], str] = "linear",
    exclude_endpoints: bool = False,
) -> List[BaseEstimator]:
    """Train regression models predicting one distance from the others.

    The idea is to take a dataset of sequential distances (e.g. produced by
    ``sequential_distances``) and build a separate scikit-learn regressor for
    each index.  Each model receives all of the distances *except* the one it
    is predicting as input features.

    Args:
        distance_matrix: array of shape ``(num_samples, num_distances)`` where
            ``num_distances == N-1`` when ``N`` is the number of keypoints.  Each
            row contains the sequential distances for a single example.
        model_factory: either a callable returning a fresh, unfitted estimator
            or a string key referring to one of the built-in regressors.  The
            following names are recognised by default:

                {}

            You can also supply a custom factory if you prefer.
        exclude_endpoints: if ``True`` the first and last distances are ignored
            when building models.  This results in ``num_distances - 2``
            regressors, which is the behaviour requested by the user (predicting
            an interior distance from the others).  Set to ``False`` to obtain a
            model for every distance.

    Returns:
        A list of fitted estimators.  When ``exclude_endpoints`` is ``True`` the
        ``i``‑th element of the returned list corresponds to original
        ``distance_matrix`` column ``i+1`` (i.e. the second sequential distance).
    """.format(
        ", ".join(sorted(DEFAULT_REGRESSORS.keys()))
    )
    if BaseEstimator is None:
        raise ImportError("scikit-learn is required for training distance models")

    # resolve string names to constructors
    if isinstance(model_factory, str):
        key = model_factory.lower()
        if key not in DEFAULT_REGRESSORS:
            raise ValueError(f"unknown regressor '{model_factory}'; valid options are: {', '.join(DEFAULT_REGRESSORS.keys())}")
        model_factory = DEFAULT_REGRESSORS[key]

    distance_matrix = np.asarray(distance_matrix)
    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix must be 2‑D (samples x distances), in this case {distance_matrix.ndim}‑D was given")

    num_samples, num_distances = distance_matrix.shape
    if num_samples < 2:
        raise ValueError("At least two samples are required to train models")

    models: List[BaseEstimator] = []
    # determine which indices to build models for
    indices = list(range(num_distances))
    if exclude_endpoints:
        if num_distances < 3:
            # nothing to train if there are fewer than three distances
            return []
        indices = list(range(1, num_distances - 1))

    for idx in indices:
        X = np.delete(distance_matrix, idx, axis=1)
        y = distance_matrix[:, idx]
        model = model_factory()
        # fit and append
        model.fit(X, y)
        models.append(model)

    return models


def predict_distances(
    models: Sequence[BaseEstimator],
    distances: Union[np.ndarray, Sequence[np.ndarray]],
    exclude_endpoints: bool = False,
) -> np.ndarray:
    """Use previously trained regressors to predict distances.

    The ``models`` sequence must have been produced by
    :func:`train_distance_models` with the same ``exclude_endpoints`` flag.

    Args:
        models: sequence of fitted estimators.
        distances: single vector (shape ``(num_distances,)``) or matrix of
            vectors (shape ``(samples, num_distances)``) containing the known
            distances.
        exclude_endpoints: same semantics as in :func:`train_distance_models`.

    Returns:
        Array of shape ``(samples, len(models))`` containing predictions for the
        requested distances.  The order of columns corresponds to the order the
        models were trained (see documentation of
        :func:`train_distance_models`).
    """
    if BaseEstimator is None:
        raise ImportError("scikit-learn is required for predicting distances")

    arr = np.asarray(distances)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("distances must be a 1‑ or 2‑D array")

    num_samples, num_distances = arr.shape
    # figure out which columns were targets during training
    indices = list(range(num_distances))
    if exclude_endpoints:
        indices = list(range(1, num_distances - 1))

    if len(models) != len(indices):
        raise ValueError("number of models does not match expected number of distances")

    preds = []
    for model, idx in zip(models, indices):
        X = np.delete(arr, idx, axis=1)
        preds.append(model.predict(X))

    return np.column_stack(preds)


def create_prediction_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    sample_ids: List[str] = None,
    metric: str = "mae"
) -> List[Dict[str, Any]]:
    """Create a sorted report comparing predictions with labels.

    Calculates error metrics per sample, sorts by error (descending),
    and returns a list of dictionaries with sample info and errors.

    Args:
        predictions: Array of shape (num_samples, num_targets) with predicted values
        labels: Array of shape (num_samples, num_targets) with true values
        sample_ids: Optional list of sample identifiers (e.g. image paths).
                    If None, uses 0-indexed sample numbers.
        metric: Error metric to sort by. Options:
                - "mae": mean absolute error per sample
                - "mse": mean squared error per sample
                - "rmse": root mean squared error per sample
                - "max": maximum absolute error per sample

    Returns:
        List of dictionaries, sorted by error (descending). Each dict contains:
            - "rank": ranking (1 = highest error)
            - "sample_id": identifier for the sample
            - "error": computed error value
            - "predictions": predicted values
            - "labels": true values
            - "differences": predictions - labels
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.shape != labels.shape:
        raise ValueError(f"predictions and labels must have same shape; got {predictions.shape} vs {labels.shape}")

    num_samples = predictions.shape[0]

    if sample_ids is None:
        sample_ids = [str(i) for i in range(num_samples)]
    elif len(sample_ids) != num_samples:
        raise ValueError(f"sample_ids length ({len(sample_ids)}) does not match predictions ({num_samples})")

    # Calculate errors
    diffs = predictions - labels
    abs_diffs = np.abs(diffs)

    if metric.lower() == "mae":
        errors = np.mean(abs_diffs, axis=1)
    elif metric.lower() == "mse":
        errors = np.mean(diffs ** 2, axis=1)
    elif metric.lower() == "rmse":
        errors = np.sqrt(np.mean(diffs ** 2, axis=1))
    elif metric.lower() == "max":
        errors = np.max(abs_diffs, axis=1)
    else:
        raise ValueError(f"unknown metric '{metric}'; options: mae, mse, rmse, max")

    # Sort descending by error
    sorted_indices = np.argsort(-errors)

    report = []
    for rank, idx in enumerate(sorted_indices, start=1):
        report.append({
            "rank": rank,
            "sample_id": sample_ids[idx],
            "error": float(errors[idx]),
            "predictions": predictions[idx].tolist(),
            "labels": labels[idx].tolist(),
            "differences": diffs[idx].tolist(),
        })

    return report
