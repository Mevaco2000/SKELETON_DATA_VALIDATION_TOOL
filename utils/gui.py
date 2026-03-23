import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
st.set_page_config(page_title="YOLO Dataset Loader", layout="wide")


def _draw_keypoints_on_img(img_bgr, image_path, label_path_map, keypoint_format="auto", visibility_threshold=None):
    """Draw all-person keypoints on img_bgr using a pre-built image_path→label_path map.
    If visibility_threshold is set, only keypoints with visibility >= threshold are drawn."""
    from utils.validation.image_validation import YPImageValidation
    _lbl = label_path_map.get(os.path.abspath(image_path))
    if _lbl and os.path.exists(_lbl):
        _all_kpts = YPImageValidation.load_all_keypoints_from_file(_lbl, keypoint_format=keypoint_format)
        _kpt_colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0),
        ]
        for _pi, _kpts in enumerate(_all_kpts):
            _ka = np.asarray(_kpts)
            if _ka.size == 0:
                continue
            if visibility_threshold is not None and _ka.ndim >= 2 and _ka.shape[1] >= 3:
                _ka = _ka[_ka[:, 2] >= visibility_threshold]
            if _ka.size == 0:
                continue
            img_bgr = YPImageValidation.draw_keypoints_on_image(
                img_bgr, _ka, color=_kpt_colors[_pi % len(_kpt_colors)], keypoint_size=4
            )
    return img_bgr


def _draw_outside_keypoints_on_img(img_bgr, image_path, label_path_map, keypoint_format, outside_kps):
    """Draw keypoints that lie outside the segmentation mask in red."""
    from utils.validation.image_validation import YPImageValidation
    _lbl = label_path_map.get(os.path.abspath(image_path))
    if not (_lbl and os.path.exists(_lbl)):
        return img_bgr
    _all_kpts = YPImageValidation.load_all_keypoints_from_file(_lbl, keypoint_format=keypoint_format)
    _h, _w = img_bgr.shape[:2]
    for _kp in outside_kps:
        _pidx = _kp.get('person_index', 0)
        _kidx = _kp.get('keypoint_index')
        if _kidx is None or _pidx >= len(_all_kpts):
            continue
        _ka = np.asarray(_all_kpts[_pidx])
        if _ka.size == 0 or _kidx >= _ka.shape[0]:
            continue
        _x = int(_ka[_kidx, 0] * _w)
        _y = int(_ka[_kidx, 1] * _h)
        cv2.circle(img_bgr, (_x, _y), 4, (0, 0, 255), -1)
    return img_bgr


def _overlay_mask_on_img(img_bgr, mask):
    """Blend a binary uint8 mask (0/255, HxW) as a semi-transparent green overlay."""
    if mask is None:
        return img_bgr
    _mask_resized = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    _green = np.zeros_like(img_bgr)
    _green[:, :, 1] = 255  # green channel
    _alpha = (_mask_resized > 0).astype(np.float32) * 0.35
    _alpha3 = _alpha[:, :, np.newaxis]
    return np.clip(img_bgr.astype(np.float32) * (1 - _alpha3) + _green.astype(np.float32) * _alpha3, 0, 255).astype(np.uint8)


def _draw_lbp_anomaly_kpt_on_img(img_bgr, image_path, label_path_map, keypoint_format, record, patch_size=32):
    """Draw all keypoints and highlight the worst-scoring LBP anomaly keypoint in red.
    Returns (annotated_img, worst_keypoint_index_or_None)."""
    from utils.validation.image_validation import YPImageValidation
    img_bgr = _draw_keypoints_on_img(img_bgr, image_path, label_path_map, keypoint_format)
    _lbl = label_path_map.get(os.path.abspath(image_path))
    if not (_lbl and os.path.exists(_lbl)):
        return img_bgr, None
    _all_kpts = YPImageValidation.load_all_keypoints_from_file(_lbl, keypoint_format=keypoint_format)
    _pidx = record.get('person_index', 0)
    if not _all_kpts or _pidx >= len(_all_kpts):
        return img_bgr, None
    _ka = np.asarray(_all_kpts[_pidx], dtype=float)
    if _ka.size == 0 or _ka.ndim < 2:
        return img_bgr, None
    _scores = np.asarray(record.get('keypoint_anomaly_scores', []), dtype=float)
    if _scores.size == 0 or np.all(np.isnan(_scores)):
        return img_bgr, None
    _worst_kp_idx = int(np.nanargmax(_scores))
    if _worst_kp_idx >= _ka.shape[0]:
        return img_bgr, None
    _h, _w = img_bgr.shape[:2]
    _x = int(_ka[_worst_kp_idx, 0] * _w)
    _y = int(_ka[_worst_kp_idx, 1] * _h)
    _half = patch_size // 2
    _x1, _y1 = max(0, _x - _half), max(0, _y - _half)
    _x2, _y2 = min(_w - 1, _x + _half), min(_h - 1, _y + _half)
    cv2.rectangle(img_bgr, (_x1, _y1), (_x2, _y2), (0, 0, 255), 2)
    cv2.circle(img_bgr, (_x, _y), 5, (0, 0, 255), -1)
    return img_bgr, _worst_kp_idx


def _draw_worst_distance_on_img(img_bgr, image_path, label_path_map, keypoint_format, item):
    """Draw keypoints plus the worst-error distance as a coloured line.
    Returns (annotated_img, worst_connection_tuple_or_None, worst_error_or_None)."""
    from utils.validation.image_validation import YPImageValidation
    img_bgr = _draw_keypoints_on_img(img_bgr, image_path, label_path_map, keypoint_format)

    _lbl = label_path_map.get(os.path.abspath(image_path))
    if not (_lbl and os.path.exists(_lbl)):
        return img_bgr, None, None

    _all_kpts = YPImageValidation.load_all_keypoints_from_file(_lbl, keypoint_format=keypoint_format)
    if not _all_kpts:
        return img_bgr, None, None
    # Use person 0 keypoints for distance drawing
    _ka = np.asarray(_all_kpts[0], dtype=float)
    if _ka.size == 0:
        return img_bgr, None, None

    _per_err = np.asarray(item.get('per_distance_errors', []), dtype=float)
    _dist_idx = item.get('distance_indices')
    _conns = item.get('distance_connections')  # list of (a,b) or None
    if _per_err.size == 0 or _dist_idx is None:
        return img_bgr, None, None

    # Find argmax ignoring NaN
    if np.all(np.isnan(_per_err)):
        return img_bgr, None, None
    _worst_pos = int(np.nanargmax(_per_err))
    _worst_err = float(_per_err[_worst_pos])
    _worst_col = int(np.asarray(_dist_idx)[_worst_pos])

    if _conns is not None:
        if _worst_col < len(_conns):
            _kp_a, _kp_b = int(_conns[_worst_col][0]), int(_conns[_worst_col][1])
        else:
            return img_bgr, None, None
    else:
        _kp_a, _kp_b = _worst_col, _worst_col + 1

    n_kpts = _ka.shape[0]
    if _kp_a >= n_kpts or _kp_b >= n_kpts:
        return img_bgr, None, None

    _h, _w = img_bgr.shape[:2]
    _xa, _ya = int(_ka[_kp_a, 0] * _w), int(_ka[_kp_a, 1] * _h)
    _xb, _yb = int(_ka[_kp_b, 0] * _w), int(_ka[_kp_b, 1] * _h)

    # Draw thick orange line for the worst distance
    cv2.line(img_bgr, (_xa, _ya), (_xb, _yb), (0, 128, 255), 3)
    # Redraw endpoints so they are visible on top
    cv2.circle(img_bgr, (_xa, _ya), 6, (0, 64, 255), -1)
    cv2.circle(img_bgr, (_xb, _yb), 6, (0, 64, 255), -1)
    return img_bgr, (_kp_a, _kp_b), _worst_err

# ================= SIDEBAR =================

st.sidebar.title("YOLO Dataset Loader")

st.sidebar.markdown(
    "<b>Paste the full path to train.txt or data.yaml:</b>",
    unsafe_allow_html=True
)

user_path = st.sidebar.text_area(
    "File path",
    value="",
    height=60,
    key="user_dataset_path"
)


def pretty_path(path):
    import re
    return re.sub(r'(?<!:)[/\\]', ' >> ', path)


if user_path.strip():
    st.sidebar.code(pretty_path(user_path.strip()), language=None)

with st.sidebar.expander("Expected dataset format", expanded=False):
    st.markdown("""
**Accepted entry files:**
- `train.txt` — list of absolute image paths (one per line)
- `data.yaml` — YOLO-style config with a `train:` key pointing to `train.txt`

**Directory layout:**
```
dataset/
├── images/
│   ├── img001.jpg
│   └── ...
└── labels/
    ├── img001.txt
    └── ...
```

**Label file format** (YOLO pose, `.txt`):
```
<class> <cx> <cy> <w> <h> \\
  <kx1> <ky1> <v1> ... <kxN> <kyN> <vN>
```
All values normalized to `[0, 1]`.  
Visibility flags: `0` = out of frame, `1` = occluded, `2` = visible.

Multiple persons per image are supported (one row per person).
""")

with st.sidebar.expander("Dataset Tools", expanded=False):
    flatten_tool = st.radio(
        "Select tool",
        ["Flatten CVAT Export", "Flatten Split Dataset"],
        key="flatten_tool_choice"
    )

    st.divider()

    if flatten_tool == "Flatten CVAT Export":
        st.markdown("**Flatten nested CVAT export into `images/` + `labels/` + `train.txt`**")
        cvat_input_root = st.text_input(
            "Input root (CVAT export folder)",
            key="cvat_input_root",
            placeholder="C:/datasets/cvat_export"
        )
        cvat_output_root = st.text_input(
            "Output root (destination folder)",
            key="cvat_output_root",
            placeholder="C:/datasets/flat_output"
        )
        cvat_copy_yaml = st.checkbox("Copy data.yaml", value=True, key="cvat_copy_yaml")

        if st.button("Run Flatten CVAT", key="btn_flatten_cvat", use_container_width=True, type="primary"):
            if cvat_input_root.strip() and cvat_output_root.strip():
                try:
                    from utils.datasets import flatten_cvat_yolo_pose
                    flatten_cvat_yolo_pose(
                        input_root=cvat_input_root.strip(),
                        output_root=cvat_output_root.strip(),
                        copy_data_yaml=cvat_copy_yaml
                    )
                    st.session_state.flatten_cvat_result = ("ok", f"Done → {cvat_output_root.strip()}")
                except Exception as exc:
                    st.session_state.flatten_cvat_result = ("err", str(exc))
            else:
                st.session_state.flatten_cvat_result = ("warn", "Fill in both paths.")

        result_cvat = st.session_state.get("flatten_cvat_result")
        if result_cvat:
            if result_cvat[0] == "ok":
                st.success(result_cvat[1])
            elif result_cvat[0] == "err":
                st.error(result_cvat[1])
            else:
                st.warning(result_cvat[1])

    else:  # Flatten Split Dataset
        st.markdown("**Flatten `images/train` + `images/val` into flat `images/` + `train.txt`**")
        split_root = st.text_input(
            "Dataset root",
            key="split_dataset_root",
            placeholder="C:/datasets/my_split_dataset"
        )
        split_splits_raw = st.text_input(
            "Splits (comma-separated, empty = auto-detect)",
            key="split_splits",
            placeholder="train, val"
        )
        col_sd1, col_sd2 = st.columns(2)
        with col_sd1:
            split_images_dir = st.text_input("Images dir", value="images", key="split_images_dir")
        with col_sd2:
            split_labels_dir = st.text_input("Labels dir", value="labels", key="split_labels_dir")
        split_output_root = st.text_input(
            "Output directory (empty = in-place)",
            key="split_output_root",
            placeholder="C:/datasets/flat_output (optional)"
        )
        split_remove_dirs = st.checkbox("Remove empty split dirs", value=True, key="split_remove_dirs",
                                        help="Ignored when output directory is set")

        if st.button("Run Flatten Split", key="btn_flatten_split", use_container_width=True, type="primary"):
            if split_root.strip():
                try:
                    from utils.datasets import flatten_split_yolo_pose
                    splits_list = (
                        [s.strip() for s in split_splits_raw.split(",") if s.strip()]
                        if split_splits_raw.strip() else None
                    )
                    train_txt_path = flatten_split_yolo_pose(
                        dataset_root=split_root.strip(),
                        splits=splits_list,
                        images_dir=split_images_dir.strip() or "images",
                        labels_dir=split_labels_dir.strip() or "labels",
                        remove_split_dirs=split_remove_dirs,
                        output_root=split_output_root.strip() or None,
                    )
                    st.session_state.flatten_split_result = ("ok", f"Done → {train_txt_path}")
                except Exception as exc:
                    st.session_state.flatten_split_result = ("err", str(exc))
            else:
                st.session_state.flatten_split_result = ("warn", "Provide dataset root path.")

        result_split = st.session_state.get("flatten_split_result")
        if result_split:
            if result_split[0] == "ok":
                st.success(result_split[1])
            elif result_split[0] == "err":
                st.error(result_split[1])
            else:
                st.warning(result_split[1])

if "page" not in st.session_state:
    st.session_state.page = "main"


page = st.sidebar.radio(
    "Navigation",
    ["main", "browse", "validate"],
    format_func=lambda x: {
        "main": "Upload",
        "browse": "Browse dataset",
        "validate": "Validate labels"
    }[x],
    key="sidebar_nav"
)

st.session_state.page = page


# ================= DATASET CACHE =================

@st.cache_resource
def load_dataset(dataset_path):
    from utils.datasets import YOLOPoseDataset
    return YOLOPoseDataset(dataset_path)


# ================= MAIN PAGE =================

if st.session_state.page == "main":

    st.markdown(
        """
        <div style='display:flex;justify-content:center;align-items:center;height:60vh;'>
            <div style='width:400px;padding:32px;border-radius:16px;background:#f8f9fa;box-shadow:0 2px 8px #0001;'>
                <h3 style='text-align:center;'>Select a <code>train.txt</code> or <code>data.yaml</code> file in the sidebar</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================= BROWSE PAGE =================

elif st.session_state.page == "browse":

    st.header("Dataset Browser")

    from utils.validation import YPImageValidation

    if not user_path.strip():
        st.warning("Paste the path to train.txt or data.yaml in the sidebar.")
        st.stop()

    if not os.path.exists(user_path.strip()):
        st.warning("The provided path does not exist on the server.")
        st.stop()

    dataset_path = user_path.strip()

    if st.button("Reload Dataset", key="btn_reload_dataset"):
        load_dataset.clear()
        st.rerun()

    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

    if len(dataset) == 0:
        st.error(
            f"Dataset contains no valid images.\n\n"
            f"Path used: `{dataset_path}`\n\n"
            f"Click **Reload Dataset** above to force a fresh load."
        )
        st.stop()

    # ================= SESSION STATE =================

    if "browse_idx" not in st.session_state:
        st.session_state.browse_idx = 0

    max_idx = len(dataset) - 1


    # ================= CALLBACKS =================

    def prev_image():
        if st.session_state.browse_idx > 0:
            st.session_state.browse_idx -= 1


    def next_image():
        if st.session_state.browse_idx < max_idx:
            st.session_state.browse_idx += 1


    # ================= SLIDER =================

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        idx = st.slider(
            "Select image",
            0,
            max_idx,
            key="browse_idx"
        )

        st.markdown(f"**Image {idx+1} / {len(dataset)}**")

    # ================= VISIBILITY FILTER =================
    
    st.divider()
    
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        show_all_keypoints = st.checkbox(
            "Show all keypoints",
            value=False,
            help="If unchecked: show only visible keypoints (visibility >= threshold)"
        )
    
    with col_filter2:
        visibility_threshold = st.slider(
            "Visibility threshold",
            min_value=0.0,
            max_value=2.0,
            value=2.0,
            step=0.1,
            help="0=all, 1=hidden or visible, 2=visible only",
            disabled=show_all_keypoints
        )

    keypoint_radius = st.slider(
        "Keypoint circle size",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Radius of the keypoint circles drawn on the image",
    )
    
    with st.expander("Visibility levels"):
        st.markdown("""
        - **0**: Out of frame
        - **1**: Hidden / Occluded
        - **2**: Visible
        
        **Examples:**
        - Slider at 0.0 → Show all keypoints
        - Slider at 1.0 → Show hidden or visible (hide out-of-frame)
        - Slider at 2.0 → Show only visible keypoints
        """)
    
    st.divider()


    # ================= IMAGE DISPLAY =================

    try:

        sample = dataset[idx]

        image = sample["image"]
        all_keypoints = sample.get("all_keypoints", [sample.get("keypoints")])

        if image is None or not all_keypoints:
            raise ValueError("Image or keypoints missing")

        # Colors for multiple persons
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light Green
            (255, 128, 0),  # Orange
        ]

        vis_img = image.copy()
        
        # Draw all persons with different colors
        for person_id, keypoints in enumerate(all_keypoints):
            if keypoints is None or len(keypoints) == 0:
                continue
            
            # Filter keypoints by visibility if needed
            kpts_to_draw = keypoints
            if not show_all_keypoints and keypoints.shape[1] >= 3:
                # Filter to visible keypoints (visibility >= threshold)
                visible_mask = keypoints[:, 2] >= visibility_threshold
                kpts_to_draw = keypoints[visible_mask]
            
            if len(kpts_to_draw) == 0:
                continue
                
            color = colors[person_id % len(colors)]
            vis_img = YPImageValidation.draw_keypoints_on_image(
                vis_img,
                kpts_to_draw,
                color=color,
                keypoint_size=keypoint_radius
            )

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        TARGET_WIDTH = 800
        TARGET_HEIGHT = 600

        vis_img = cv2.resize(vis_img, (TARGET_WIDTH, TARGET_HEIGHT))

        col_left, col_center, col_right = st.columns([1,2,1])

        with col_center:
            st.image(
                vis_img,
                caption=f"{os.path.basename(sample['image_path'])} ({len(all_keypoints)} persons)"
            )

    except Exception as e:

        st.error(f"Visualization error: {e}")


    # ================= ARROWS =================

    col_left, col_spacer, col_right = st.columns([1,6,1])

    with col_left:
        st.button("<", on_click=prev_image, key="btn_prev")

    with col_right:
        st.button(">", on_click=next_image, key="btn_next")



# ================= VALIDATE PAGE =================

elif st.session_state.page == "validate":

    st.header("Label Validation Pipeline")

    if not user_path.strip():
        st.warning("Paste the path to train.txt or data.yaml in the sidebar.")
        st.stop()

    if not os.path.exists(user_path.strip()):
        st.warning("The provided path does not exist on the server.")
        st.stop()

    # ================= VALIDATION STAGES =================
    
    if "validation_stage" not in st.session_state:
        st.session_state.validation_stage = 0
    
    stages = [
        "Select Validation Type",
        "Configure & Run",
        "View Results"
    ]
    
    # Progress bar
    col_progress = st.columns(len(stages))
    for i, (col, stage) in enumerate(zip(col_progress, stages)):
        with col:
            if i == st.session_state.validation_stage:
                st.write(f"**> {stage}**")
            elif i < st.session_state.validation_stage:
                st.write(f"[done] {stage}")
            else:
                st.write(f"[ ] {stage}")
    
    st.divider()
    
    # ================= STAGE 0: SELECT VALIDATION TYPE =================
    
    if st.session_state.validation_stage == 0:
        
        st.subheader("Select Validation Type")
        
        validation_type = st.radio(
            "What would you like to validate?",
            ["LBP Anomalies", "Segmentation Evaluation", "Duplicate Images", "Distance Anomalies", "Full Pipeline Validation"],
            captions=[
                "Detect LBP anomalies grouped by image similarity",
                "Evaluate keypoints against segmentation masks using YOLO models",
                "Find duplicate or near-duplicate images using CLIP embeddings",
                "Train models and detect anomalies in sequential distances",
                "Run all validation checks sequentially for comprehensive analysis"
            ]
        )
        
        st.session_state.validation_type = validation_type
        
        col_next = st.columns([0.8, 0.2])
        with col_next[1]:
            if st.button("Next →", use_container_width=True):
                st.session_state.validation_stage = 1
                st.rerun()
    
    # ================= STAGE 1: CONFIGURE & RUN =================
    
    elif st.session_state.validation_stage == 1:
        
        st.subheader(f"Configure: {st.session_state.validation_type}")
        
        dataset_path = user_path.strip()
        st.session_state.dataset_path = dataset_path  # Store in session state
        
        col_back, col_config = st.columns([0.2, 0.8])
        
        with col_config:
            if st.session_state.validation_type == "LBP Anomalies":
                _lbp_cfg = st.session_state.get("config", {})
                patch_size = st.slider("LBP Patch Size", 4, 256, _lbp_cfg.get("patch_size", 4), step=2,
                    help="Size (in pixels) of the square patch around each keypoint used to compute the LBP texture descriptor. Larger values capture more context but are slower.")
                visibility_threshold = st.slider("Visibility Threshold", 0.0, 2.0, _lbp_cfg.get("visibility_threshold", 2.0), step=0.1,
                    help="Minimum keypoint visibility flag to include in analysis. 0 = all keypoints, 1 = hidden or visible, 2 = fully visible only (YOLO convention).")
                contamination = st.slider("Contamination (anomaly rate)", 0.01, 0.5, _lbp_cfg.get("contamination", 0.1), step=0.01,
                    help="Expected fraction of anomalous samples in the dataset. Used by the Isolation Forest classifier to set its decision threshold. Lower values = stricter anomaly detection.")
                similarity_threshold = st.slider("Similarity Threshold", 0.80, 1.0, _lbp_cfg.get("similarity_threshold", 0.95), step=0.01,
                    help="Minimum cosine similarity between CLIP image embeddings to consider two images part of the same group. Higher values = smaller, more homogeneous groups.")
                min_group_size = st.slider("Min Group Size", 5, 50, _lbp_cfg.get("min_group_size", 10), step=1,
                    help="Minimum number of images required to train an Isolation Forest model for a group. Groups smaller than this threshold are skipped.")
                st.session_state.config = {
                    "patch_size": patch_size,
                    "visibility_threshold": visibility_threshold,
                    "contamination": contamination,
                    "similarity_threshold": similarity_threshold,
                    "min_group_size": min_group_size
                }
                
            elif st.session_state.validation_type == "Segmentation Evaluation":
                _seg_model_options = ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg", "Custom..."]
                model_name_sel = st.selectbox("Segmentation Model", _seg_model_options,
                    help="YOLO segmentation model used to generate person masks. Larger models (l, x) are more accurate but slower. Choose 'Custom...' to provide your own model path.")
                if model_name_sel == "Custom...":
                    model_name_custom = st.text_input("Custom model path", value="",
                        placeholder="C:/models/my_seg_model.pt",
                        help="Absolute path to a custom YOLO segmentation .pt file.")
                    model_name = model_name_custom.strip() if model_name_custom.strip() else None
                    if not model_name:
                        st.warning("Provide a path to the custom model.")
                else:
                    model_name = model_name_sel
                seg_score_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05,
                    help="Minimum confidence score for a detected person mask to be accepted. Lower values include less certain detections.")
                seg_target_class = st.text_input("Target Class", value="person",
                    help="Class name to segment. Use 'person' for human pose datasets.")
                visibility_threshold = st.slider("Visibility Threshold", 0.0, 2.0, 2.0, step=0.1,
                    help="Minimum keypoint visibility flag to include in analysis. 0 = all keypoints, 1 = hidden or visible, 2 = fully visible only (YOLO convention).")
                st.session_state.config = {
                    "model_name": model_name,
                    "seg_score_threshold": seg_score_threshold,
                    "seg_target_class": seg_target_class.strip() or "person",
                    "visibility_threshold": visibility_threshold,
                }
                
            elif st.session_state.validation_type == "Duplicate Images":
                k_neighbors = st.slider("Number of Neighbors", 1, 20, 5,
                    help="Number of nearest neighbors to retrieve per image when searching for duplicates. A higher value finds more potential duplicates but increases computation time.")
                threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.99, step=0.01,
                    help="Minimum cosine similarity between CLIP embeddings for two images to be considered near-duplicates. 1.0 = exact duplicates only, lower values also catch visually similar images.")
                st.session_state.config = {"k_neighbors": k_neighbors, "threshold": threshold}
                
            elif st.session_state.validation_type == "Distance Anomalies":
                model_type = st.selectbox("Model Type", ["hist_gradient_boosting", "random_forest", "knn"],
                    help="Regression model used to predict expected keypoint distances. hist_gradient_boosting is fastest and most accurate; knn is simplest.")
                threshold_percentile = st.slider("Anomaly Threshold (percentile)", 50, 100, 95,
                    help="Images whose prediction error exceeds this percentile of all errors are flagged as anomalies. 95 = top 5% worst errors are anomalies.")
                st.session_state.config = {"model_type": model_type, "threshold_percentile": threshold_percentile}
            
            elif st.session_state.validation_type == "Full Pipeline Validation":
                st.write("**Select which validations to run:**")
                col1, col2 = st.columns(2)
                with col1:
                    run_lbp = st.checkbox("LBP Anomalies", value=True)
                    run_segmentation = st.checkbox("Segmentation Evaluation", value=True)
                with col2:
                    run_duplicates = st.checkbox("Duplicate Images", value=True)
                    run_distance = st.checkbox("Distance Anomalies", value=True)
                
                st.divider()
                st.write("**LBP Anomalies Parameters:**")
                col_lbp1, col_lbp2 = st.columns(2)
                with col_lbp1:
                    lbp_patch_size = st.slider("LBP Patch Size", 4, 256, 4, step=2, key="pipeline_lbp_patch",
                        help="Size (in pixels) of the square patch around each keypoint used to compute the LBP texture descriptor. Larger values capture more context but are slower.")
                    lbp_contamination = st.slider("Contamination", 0.01, 0.5, 0.1, step=0.01, key="pipeline_lbp_cont",
                        help="Expected fraction of anomalous samples in the dataset. Used by the Isolation Forest classifier to set its decision threshold. Lower values = stricter anomaly detection.")
                with col_lbp2:
                    lbp_similarity = st.slider("Similarity Threshold", 0.80, 1.0, 0.95, step=0.01, key="pipeline_lbp_sim",
                        help="Minimum cosine similarity between CLIP image embeddings to consider two images part of the same group. Higher values = smaller, more homogeneous groups.")
                    lbp_min_group = st.slider("Min Group Size", 5, 50, 10, step=1, key="pipeline_lbp_group",
                        help="Minimum number of images required to train an Isolation Forest model for a group. Groups smaller than this threshold are skipped.")
                
                st.write("**Segmentation Parameters:**")
                _pl_seg_options = ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg", "Custom..."]
                seg_model_sel = st.selectbox("Segmentation Model", _pl_seg_options, key="pipeline_seg_model",
                    help="YOLO segmentation model used to generate person masks. Larger models (l, x) are more accurate but slower. Choose 'Custom...' to provide your own model path.")
                if seg_model_sel == "Custom...":
                    seg_model_custom = st.text_input("Custom model path", value="", key="pipeline_seg_custom_path",
                        placeholder="C:/models/my_seg_model.pt",
                        help="Absolute path to a custom YOLO segmentation .pt file.")
                    seg_model = seg_model_custom.strip() if seg_model_custom.strip() else None
                    if not seg_model:
                        st.warning("Provide a path to the custom model.")
                else:
                    seg_model = seg_model_sel
                col_seg1, col_seg2 = st.columns(2)
                with col_seg1:
                    seg_score_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05, key="pipeline_seg_score",
                        help="Minimum confidence score for a detected person mask to be accepted. Lower values include less certain detections.")
                with col_seg2:
                    seg_target_class = st.text_input("Target Class", value="person", key="pipeline_seg_class",
                        help="Class name to segment. Use 'person' for human pose datasets.")
                
                st.write("**General Parameters:**")
                visibility_threshold = st.slider("Visibility Threshold", 0.0, 2.0, 2.0, step=0.1, key="pipeline_vis",
                    help="Minimum keypoint visibility flag to include in analysis. 0 = all keypoints, 1 = hidden or visible, 2 = fully visible only (YOLO convention).")
                duplicates_threshold = st.slider("Duplicate Similarity Threshold", 0.0, 1.0, 0.99, step=0.01, key="pipeline_dup_sim",
                    help="Minimum cosine similarity between CLIP embeddings for two images to be considered near-duplicates. 1.0 = exact duplicates only, lower values also catch visually similar images.")
                distance_threshold = st.slider("Distance Anomaly Threshold (percentile)", 50, 100, 95, key="pipeline_dist_perc",
                    help="Images whose prediction error exceeds this percentile of all errors are flagged as anomalies. 95 = top 5% worst errors are anomalies.")
                
                st.session_state.config = {
                    "run_lbp": run_lbp,
                    "run_segmentation": run_segmentation,
                    "run_duplicates": run_duplicates,
                    "run_distance": run_distance,
                    "lbp_patch_size": lbp_patch_size,
                    "lbp_contamination": lbp_contamination,
                    "lbp_similarity": lbp_similarity,
                    "lbp_min_group": lbp_min_group,
                    "seg_model": seg_model,
                    "seg_score_threshold": seg_score_threshold,
                    "seg_target_class": seg_target_class.strip() or "person",
                    "visibility_threshold": visibility_threshold,
                    "duplicates_threshold": duplicates_threshold,
                    "distance_threshold": distance_threshold
                }
        
        st.divider()
        
        col_back, col_spacer, col_run = st.columns([0.2, 0.6, 0.2])
        
        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.validation_stage = 0
                st.rerun()
        
        with col_run:
            if st.button("Run Validation", use_container_width=True, type="primary"):
                st.session_state.validation_stage = 2
                st.rerun()
    
    # ================= STAGE 2: VIEW RESULTS =================
    
    elif st.session_state.validation_stage == 2:
        
        st.subheader(f"Results: {st.session_state.validation_type}")
        
        dataset_path = st.session_state.get("dataset_path", user_path.strip())
        
        with st.spinner("Loading dataset..."):
            dataset = load_dataset(dataset_path)

        # Build image_path → label_path map for fast keypoint lookup in rankings
        _lbl_map = {os.path.abspath(_ip): _lp for _ip, _lp in dataset._valid_pairs}
        _kpt_fmt = dataset.keypoint_format
        
        validation_type = st.session_state.validation_type
        config = st.session_state.config
        
        try:
            from utils.validation import YPSetValidation
            validator = YPSetValidation(dataset)
            
            if validation_type == "LBP Anomalies":
                _lbp_cache_key = (
                    dataset_path,
                    config["patch_size"],
                    config["visibility_threshold"],
                    config["contamination"],
                    config["similarity_threshold"],
                    config["min_group_size"],
                )
                if st.session_state.get("_lbp_cache_key") != _lbp_cache_key:
                    with st.status("Processing LBP anomalies...", expanded=True) as status:
                        status.write("Computing LBP anomalies by embedding groups...")
                        results = validator.predict_lbp_anomalies_by_embedding_groups(
                            patch_size=config["patch_size"],
                            visibility_threshold=config["visibility_threshold"],
                            contamination=config["contamination"],
                            similarity_threshold=config["similarity_threshold"],
                            min_group_size=config["min_group_size"],
                            verbose=True
                        )
                        status.update(label="LBP anomalies computed", state="complete")
                    st.session_state["_lbp_results"] = results
                    st.session_state["_lbp_cache_key"] = _lbp_cache_key
                else:
                    results = st.session_state["_lbp_results"]
                
                records = results.get('records', [])
                groups = results.get('groups', [])
                trained_groups = results.get('trained_group_ids', [])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalous Records", sum(1 for r in records if r.get('is_anomaly', False)))
                with col2:
                    st.metric("Total Records", len(records))
                with col3:
                    st.metric("Groups Trained", len(trained_groups))
                
                st.info(f"LBP anomalies computed across {len(trained_groups)} image similarity groups")

                if not records:
                    all_groups = results.get('all_similarity_groups', [])
                    st.warning(
                        f"No records to display — no similarity group reached the **Min Group Size** threshold ({config['min_group_size']} images). "
                        f"Found {len(all_groups)} group(s) in total. "
                        f"Try lowering **Min Group Size** or **Similarity Threshold** in the configuration."
                    )
                else:
                    _lbp_n_show = st.number_input("Show top N", min_value=1, max_value=max(1, len(records)), value=max(1, min(10, len(records))), step=1, key="lbp_n_show")
                    st.subheader(f"Top {_lbp_n_show} by Mean Anomaly Score (all images in trained groups)")
                    sorted_records = sorted(records, key=lambda x: x.get('mean_anomaly_score', 0), reverse=True)
                    for i, record in enumerate(sorted_records[:_lbp_n_show], 1):
                        col_rank, col_img, col_content = st.columns([0.05, 0.2, 0.75])
                        with col_rank:
                            st.write(f"**#{i}**")
                        with col_img:
                            _prev = cv2.imread(record['image_path']) if os.path.exists(record['image_path']) else None
                            if _prev is not None:
                                _prev, _worst_kp = _draw_lbp_anomaly_kpt_on_img(_prev, record['image_path'], _lbl_map, _kpt_fmt, record, patch_size=config["patch_size"])
                                st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                        with col_content:
                            _is_anom = record.get('is_anomaly', False)
                            _anom_label = " 🔴 ANOMALY" if _is_anom else ""
                            st.write(f"**{os.path.basename(record['image_path'])}** (Person {record['person_index']}){_anom_label}")
                            st.write(f"Mean Anomaly Score: {record.get('mean_anomaly_score', float('nan')):.4f} | Group: {record['group_id']}")
                            _worst_kp = None
                            _sc = np.asarray(record.get('keypoint_anomaly_scores', []), dtype=float)
                            if _sc.size > 0 and not np.all(np.isnan(_sc)):
                                _worst_kp = int(np.nanargmax(_sc))
                            if _worst_kp is not None:
                                st.write(f"Worst keypoint: **#{_worst_kp}** (score {float(np.nanmax(_sc)):.4f})")
                
            elif validation_type == "Segmentation Evaluation":
                dataset_size = len(dataset)
                
                with st.status("Processing segmentation and evaluation...", expanded=True) as status:
                    status.write("Step 1/3: Segmenting dataset with model...")
                    
                    segmented = validator.segment_dataset_with_model(
                        model_name=config["model_name"],
                        target_class=config.get("seg_target_class", "person"),
                        score_threshold=config.get("seg_score_threshold", 0.5),
                        verbose=True
                    )
                    status.write(f"Segmented {len(segmented['masks'])} images")
                    
                    status.write("Step 2/3: Evaluating keypoints against masks...")
                    
                    results_mask = validator.evaluate_dataset_keypoints_against_masks(
                        masks=segmented["masks"],
                        image_paths=segmented["image_paths"],
                        visibility_threshold=config["visibility_threshold"],
                        verbose=True
                    )
                    status.write(f"Evaluated keypoints for {len(results_mask['image_paths'])} images")
                    
                    status.write("Step 3/3: Analyzing visible keypoints outside masks...")
                    
                    outside_keypoints = validator.collect_visible_keypoints_outside_masks(results_mask)
                    status.write(f"Analyzed {len(outside_keypoints)} keypoints outside masks")
                    status.update(label="All evaluations complete", state="complete")

                # Build mask lookup: abs image_path -> mask array
                _seg_mask_by_path = {
                    os.path.abspath(_p): _m
                    for _p, _m in zip(segmented["image_paths"], segmented["masks"])
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images Evaluated", len(results_mask["image_paths"]))
                with col2:
                    st.metric("Keypoints Outside Masks", len(outside_keypoints))
                with col3:
                    st.metric("Visibility Threshold", config["visibility_threshold"])
                
                st.info(f"Evaluated keypoints against segmentation masks using {config['model_name']}")
                
                with st.expander("View Evaluation Summary"):
                    st.write(f"**Total images processed**: {len(results_mask['image_paths'])}")
                    st.write(f"**Model used**: {config['model_name']}")
                    st.write(f"**Visibility threshold**: {results_mask.get('visibility_threshold', config['visibility_threshold'])}")
                    st.write(f"**Visible keypoints outside masks**: {len(outside_keypoints)}")
                    
                    if len(outside_keypoints) > 0:
                        st.subheader("Images with Keypoints Outside Masks")
                        # Group by image path
                        _img_kp_map = {}
                        for _kp in outside_keypoints:
                            _ip = _kp.get('image_path', 'N/A')
                            _img_kp_map.setdefault(_ip, []).append(_kp)
                        # Sort images by number of outside keypoints descending
                        _sorted_imgs = sorted(_img_kp_map.items(), key=lambda x: len(x[1]), reverse=True)
                        _seg_n_show = st.number_input("Show top N", min_value=1, max_value=max(1, len(_sorted_imgs)), value=max(1, min(20, len(_sorted_imgs))), step=1, key="seg_n_show")
                        for i, (_ip, _kps) in enumerate(_sorted_imgs[:_seg_n_show], 1):
                            col_rank, col_img, col_content = st.columns([0.05, 0.2, 0.75])
                            with col_rank:
                                st.write(f"**#{i}**")
                            with col_img:
                                _prev = cv2.imread(_ip) if os.path.exists(_ip) else None
                                if _prev is not None:
                                    _prev = _overlay_mask_on_img(_prev, _seg_mask_by_path.get(os.path.abspath(_ip)))
                                    _prev = _draw_keypoints_on_img(_prev, _ip, _lbl_map, _kpt_fmt, visibility_threshold=2.0)
                                    _prev = _draw_outside_keypoints_on_img(_prev, _ip, _lbl_map, _kpt_fmt, _kps)
                                    st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                            with col_content:
                                st.write(f"**{os.path.basename(_ip)}** — {len(_kps)} keypoint(s) outside")
                                _kp_ids = [f"{_k.get('keypoint_index','?')}(p{_k.get('person_index','?')})" for _k in _kps]
                                st.write(f"Keypoints (red): {', '.join(_kp_ids)}") 
                
            elif validation_type == "Duplicate Images":
                with st.status("Computing CLIP embeddings...", expanded=True) as status:
                    status.write("Processing images...")
                    
                    embeddings, filenames = validator.get_embeddings(device=None)
                    status.update(label="Embeddings computed", state="complete")
                
                st.metric("Images Embedded", len(filenames))
                st.info("CLIP embeddings computed successfully.")
                
            elif validation_type == "Distance Anomalies":
                with st.status("Training and detecting anomalies...", expanded=True) as status:
                    status.write("Step 1/2: Training distance models...")
                    
                    models, valid_indices = validator.train_distance_models(
                        model_factory=config["model_type"],
                        verbose=True
                    )
                    status.write(f"Trained {len(models)} models")
                
                if len(models) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Models Trained", len(models))
                    with col2:
                        st.metric("Distance Columns", len(valid_indices))
                    
                    with st.status("Testing models...", expanded=False) as test_status:
                        test_status.write("Detecting anomalies...")
                        
                        _dist_result = validator.predict_distance_anomalies(
                            models,
                            valid_indices,
                            threshold_percentile=config["threshold_percentile"],
                            verbose=True
                        )
                        anomalies = _dist_result['images']
                        test_status.write(f"Found {len(anomalies)} ranked items")
                        test_status.update(label="Anomaly detection complete", state="complete")
                    
                    anomaly_count = sum(1 for a in anomalies if a['is_anomaly'])
                    st.metric("Anomalies Found (by threshold)", anomaly_count, f"({100*anomaly_count/len(anomalies):.1f}%)")
                    st.metric("Total Ranked Items", len(anomalies))
                    
                    _dist_n_show = st.number_input("Show top N", min_value=1, max_value=max(1, len(anomalies)), value=max(1, min(10, len(anomalies))), step=1, key="dist_n_show")
                    st.subheader(f"Top {_dist_n_show} Distance Anomalies by Error Score")
                    sorted_anomalies = sorted(anomalies, key=lambda x: x['error'], reverse=True)
                    for position, item in enumerate(sorted_anomalies[:_dist_n_show], 1):
                        col_rank, col_img, col_content = st.columns([0.05, 0.2, 0.75])
                        with col_rank:
                            st.write(f"**#{position}**")
                        with col_img:
                            _prev = cv2.imread(item['image_path']) if os.path.exists(item['image_path']) else None
                            if _prev is not None:
                                _prev, _wconn, _werr = _draw_worst_distance_on_img(
                                    _prev, item['image_path'], _lbl_map, _kpt_fmt, item
                                )
                                st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                        with col_content:
                            st.write(f"**{os.path.basename(item['image_path'])}**")
                            st.write(f"Mean Error: `{item['error']:.4f}`")
                            if _wconn is not None:
                                st.write(f"Worst distance: kp **{_wconn[0]}** → **{_wconn[1]}** (err `{_werr:.4f}`)") 
                else:
                    st.warning("No models trained (insufficient distance data)")
            
            elif validation_type == "Full Pipeline Validation":

                # Sub-stage keys: 0=detect_dups, 1=review_pairs, 2=run_analysis, 3=results
                if "pipeline_sub_stage" not in st.session_state:
                    st.session_state.pipeline_sub_stage = 0
                if "deleted_basenames" not in st.session_state:
                    st.session_state.deleted_basenames = set()

                sub_stage = st.session_state.pipeline_sub_stage

                # Sub-stage progress bar
                _sub_stages = ["1. Detect Duplicates", "2. Review Pairs", "3. Run Analysis", "4. Results"]
                _sub_cols = st.columns(len(_sub_stages))
                for _i, (_sc, _sn) in enumerate(zip(_sub_cols, _sub_stages)):
                    with _sc:
                        if _i == sub_stage:
                            st.write(f"**> {_sn}**")
                        elif _i < sub_stage:
                            st.write(f"[done] {_sn}")
                        else:
                            st.write(f"[ ] {_sn}")
                st.divider()

                # ---- SUB-STAGE 0: DETECT DUPLICATES ----
                if sub_stage == 0:
                    if "dup_pairs" not in st.session_state:
                        with st.status("Detecting duplicate images...", expanded=True) as _dup_detect_status:
                            _dup_detect_status.write("Computing CLIP embeddings for all images...")
                            _emb, _fnames = validator._get_embeddings_with_image_paths(device=None, verbose=False)
                            st.session_state.pipeline_embeddings = _emb
                            st.session_state.pipeline_embedding_paths = _fnames
                            _dup_detect_status.write(f"Embedded {len(_fnames)} images. Finding near-duplicate pairs...")

                            from utils.validation import YPSetValidation as _YPSetValidation
                            _raw_pairs = _YPSetValidation.find_near_duplicates(
                                _emb, _fnames,
                                threshold=config.get("duplicates_threshold", 0.99)
                            )

                            # Deduplicate: keep only (A, B), not both (A,B) and (B,A)
                            _seen_keys = set()
                            _deduped = []
                            for _a, _b, _sim in _raw_pairs:
                                _key = (min(_a, _b), max(_a, _b))
                                if _key not in _seen_keys:
                                    _seen_keys.add(_key)
                                    _deduped.append((_a, _b, _sim))

                            st.session_state.dup_pairs = _deduped
                            st.session_state.dup_images_dir = os.path.join(dataset.dataset_dir, "images")
                            st.session_state.dup_labels_dir = dataset.labels_dir
                            _dup_detect_status.update(
                                label=f"Found {len(_deduped)} duplicate pair(s)", state="complete"
                            )

                    _n_pairs = len(st.session_state.dup_pairs)
                    if _n_pairs == 0:
                        st.info("No duplicate pairs found.")
                        st.session_state.pipeline_sub_stage = 2
                        st.rerun()
                    else:
                        st.success(f"Found **{_n_pairs}** near-duplicate pair(s).")
                        _col_review, _col_skip = st.columns(2)
                        with _col_review:
                            if st.button(f"Review {_n_pairs} pairs →", key="btn_start_review", use_container_width=True):
                                st.session_state.dup_pair_idx = 0
                                st.session_state.deleted_basenames = set()
                                st.session_state.pipeline_sub_stage = 1
                                st.rerun()
                        with _col_skip:
                            if st.button("Continue to Analysis →", key="btn_dup_skip_to_analysis", use_container_width=True, type="primary"):
                                st.session_state.pipeline_sub_stage = 2
                                st.rerun()

                # ---- SUB-STAGE 1: REVIEW PAIRS ----
                elif sub_stage == 1:
                    _pairs = st.session_state.dup_pairs
                    _deleted = st.session_state.deleted_basenames
                    _images_dir = st.session_state.dup_images_dir
                    _labels_dir = st.session_state.dup_labels_dir

                    # Skip pairs where either image is already deleted
                    _idx = st.session_state.get("dup_pair_idx", 0)
                    while _idx < len(_pairs) and (_pairs[_idx][0] in _deleted or _pairs[_idx][1] in _deleted):
                        _idx += 1
                    st.session_state.dup_pair_idx = _idx

                    _total = len(_pairs)
                    _deleted_count = len(_deleted)
                    _remaining = sum(1 for _pa, _pb, _ in _pairs if _pa not in _deleted and _pb not in _deleted)

                    if _idx >= _total:
                        st.success(f"All pairs reviewed! Deleted **{_deleted_count}** image(s).")
                        if st.button("Continue to Analysis →", key="btn_review_done", use_container_width=True, type="primary"):
                            if _deleted_count > 0:
                                load_dataset.clear()
                            st.session_state.pipeline_results = None
                            st.session_state.pipeline_sub_stage = 2
                            st.rerun()
                    else:
                        _basename_a, _basename_b, _sim = _pairs[_idx]
                        st.markdown(
                            f"**Pair {_idx + 1} / {_total}** &nbsp;|&nbsp; "
                            f"Similarity: `{_sim:.4f}` &nbsp;|&nbsp; "
                            f"Remaining: {_remaining} &nbsp;|&nbsp; "
                            f"Deleted so far: {_deleted_count}"
                        )

                        _path_a = os.path.join(_images_dir, _basename_a)
                        _path_b = os.path.join(_images_dir, _basename_b)

                        _col_a, _col_b = st.columns(2)

                        with _col_a:
                            st.markdown(f"**A:** `{_basename_a}`")
                            _img_a = cv2.imread(_path_a) if os.path.exists(_path_a) else None
                            if _img_a is not None:
                                st.image(cv2.cvtColor(_img_a, cv2.COLOR_BGR2RGB), width='stretch')
                            else:
                                st.warning("Image not found on disk")
                            if st.button("Delete A", key="btn_del_a", use_container_width=True, type="primary"):
                                _lbl_a = os.path.join(_labels_dir, os.path.splitext(_basename_a)[0] + ".txt")
                                try:
                                    if os.path.exists(_path_a):
                                        os.remove(_path_a)
                                    if os.path.exists(_lbl_a):
                                        os.remove(_lbl_a)
                                    _deleted.add(_basename_a)
                                    st.session_state.deleted_basenames = _deleted
                                    st.session_state.dup_pair_idx = _idx + 1
                                except Exception as _exc:
                                    st.error(f"Delete failed: {_exc}")
                                st.rerun()

                        with _col_b:
                            st.markdown(f"**B:** `{_basename_b}`")
                            _img_b = cv2.imread(_path_b) if os.path.exists(_path_b) else None
                            if _img_b is not None:
                                st.image(cv2.cvtColor(_img_b, cv2.COLOR_BGR2RGB), width='stretch')
                            else:
                                st.warning("Image not found on disk")
                            if st.button("Delete B", key="btn_del_b", use_container_width=True, type="primary"):
                                _lbl_b = os.path.join(_labels_dir, os.path.splitext(_basename_b)[0] + ".txt")
                                try:
                                    if os.path.exists(_path_b):
                                        os.remove(_path_b)
                                    if os.path.exists(_lbl_b):
                                        os.remove(_lbl_b)
                                    _deleted.add(_basename_b)
                                    st.session_state.deleted_basenames = _deleted
                                    st.session_state.dup_pair_idx = _idx + 1
                                except Exception as _exc:
                                    st.error(f"Delete failed: {_exc}")
                                st.rerun()

                        st.divider()
                        _col_keep, _col_skip_all = st.columns(2)
                        with _col_keep:
                            if st.button("Keep Both →", key="btn_keep_both", use_container_width=True):
                                st.session_state.dup_pair_idx = _idx + 1
                                st.rerun()
                        with _col_skip_all:
                            if st.button("Skip Remaining & Continue →", key="btn_skip_remaining", use_container_width=True):
                                if _deleted_count > 0:
                                    load_dataset.clear()
                                st.session_state.pipeline_results = None
                                st.session_state.pipeline_sub_stage = 2
                                st.rerun()

                # ---- SUB-STAGE 2: RUN ANALYSIS ----
                elif sub_stage == 2:
                    if "pipeline_results" not in st.session_state or st.session_state.pipeline_results is None:
                        # Reload dataset in case images were deleted
                        with st.spinner("Loading dataset..."):
                            _fresh_dataset = load_dataset(dataset_path)
                        from utils.validation import YPSetValidation as _YPSetValidation2
                        _pipeline_validator = _YPSetValidation2(_fresh_dataset)

                        st.subheader("Running Analysis Pipeline")
                        pipeline_results = {}

                        # Step 2/3: LBP Anomalies
                        if config.get("run_lbp", True):
                            with st.status("Step 1/3: LBP Anomalies Detection", expanded=False) as _lbp_s:
                                _lbp_s.write("Computing LBP anomalies...")
                                _cached_emb = st.session_state.get("pipeline_embeddings")
                                _cached_paths = st.session_state.get("pipeline_embedding_paths")
                                _lbp_results = _pipeline_validator.predict_lbp_anomalies_by_embedding_groups(
                                    patch_size=config["lbp_patch_size"],
                                    visibility_threshold=config["visibility_threshold"],
                                    contamination=config["lbp_contamination"],
                                    similarity_threshold=config["lbp_similarity"],
                                    min_group_size=config["lbp_min_group"],
                                    verbose=True,
                                    precomputed_embeddings=_cached_emb,
                                    precomputed_image_paths=_cached_paths,
                                )
                                pipeline_results["lbp"] = _lbp_results
                                _lbp_s.update(label="LBP Anomalies detected", state="complete")

                        # Step 3/3: Segmentation Evaluation
                        if config.get("run_segmentation", True):
                            with st.status("Step 2/3: Segmentation Evaluation", expanded=False) as _seg_s:
                                _seg_s.write("Segmenting dataset...")
                                _segmented = _pipeline_validator.segment_dataset_with_model(
                                    model_name=config["seg_model"],
                                    target_class=config.get("seg_target_class", "person"),
                                    score_threshold=config.get("seg_score_threshold", 0.5),
                                    verbose=True
                                )
                                _seg_s.write("Evaluating keypoints...")
                                _results_mask = _pipeline_validator.evaluate_dataset_keypoints_against_masks(
                                    masks=_segmented["masks"],
                                    image_paths=_segmented["image_paths"],
                                    visibility_threshold=config["visibility_threshold"],
                                    verbose=True
                                )
                                _seg_s.write("Analyzing keypoints outside masks...")
                                _outside_kp = _pipeline_validator.collect_visible_keypoints_outside_masks(_results_mask)
                                pipeline_results["segmentation"] = {
                                    "segmented_images": len(_segmented["masks"]),
                                    "evaluated_images": len(_results_mask["image_paths"]),
                                    "keypoints_outside_masks": len(_outside_kp),
                                    "outside_keypoints": _outside_kp,
                                    "mask_by_path": {
                                        os.path.abspath(_p): _m
                                        for _p, _m in zip(_segmented["image_paths"], _segmented["masks"])
                                    },
                                }
                                _seg_s.update(label="Segmentation Evaluation complete", state="complete")

                        # Step 4/4: Distance Anomalies
                        if config.get("run_distance", True):
                            with st.status("Step 3/3: Distance Anomalies Detection", expanded=False) as _dist_s:
                                _dist_s.write("Training distance models...")
                                _models, _valid_idx = _pipeline_validator.train_distance_models(
                                    model_factory="hist_gradient_boosting",
                                    visibility_threshold=config["visibility_threshold"],
                                    verbose=True
                                )
                                _dist_s.write("Detecting anomalies...")
                                _dist_result = _pipeline_validator.predict_distance_anomalies(
                                    _models, _valid_idx,
                                    visibility_threshold=config["visibility_threshold"],
                                    threshold_percentile=config["distance_threshold"],
                                    verbose=True
                                )
                                _anomalies = _dist_result['images']
                                pipeline_results["distance"] = {
                                    "models_trained": len(_models),
                                    "anomalies_detected": len(_anomalies),
                                    "anomalies": _anomalies
                                }
                                _dist_s.update(label="Distance Anomalies detected", state="complete")

                        st.session_state.pipeline_results = pipeline_results
                        st.session_state.pipeline_sub_stage = 3
                        st.rerun()

                # ---- SUB-STAGE 3: RESULTS ----
                elif sub_stage == 3:
                    pipeline_results = st.session_state.pipeline_results

                    col_title, col_rerun = st.columns([0.8, 0.2])
                    with col_title:
                        _del_count = len(st.session_state.get("deleted_basenames", set()))
                        st.success(f"Pipeline Complete! ({_del_count} duplicate(s) deleted)")
                    with col_rerun:
                        if st.button("Re-run Pipeline", use_container_width=True):
                            st.session_state.pipeline_results = None
                            st.session_state.pipeline_sub_stage = 0
                            if "dup_pairs" in st.session_state:
                                del st.session_state["dup_pairs"]
                            st.session_state.deleted_basenames = set()
                            st.rerun()

                    st.divider()
                    st.subheader("Pipeline Results - Check Validation Scores")

                    col1, col2, col3 = st.columns(3)

                    # LBP Anomalies
                    if "lbp" in pipeline_results:
                        with col1:
                            if st.button("LBP Anomalies Scores", use_container_width=True):
                                st.session_state.show_lbp_scores = not st.session_state.get("show_lbp_scores", False)
                        if st.session_state.get("show_lbp_scores", False):
                            with st.expander("LBP Anomalies - Top Scores", expanded=True):
                                _lbp_rec = pipeline_results["lbp"].get('records', [])
                                if not _lbp_rec:
                                    _all_grps = pipeline_results["lbp"].get('all_similarity_groups', [])
                                    st.warning(
                                        f"No LBP records to display — no similarity group reached the minimum group size threshold. "
                                        f"Found {len(_all_grps)} group(s) in total. "
                                        f"Try lowering **Min Group Size** or **Similarity Threshold** in the pipeline configuration."
                                    )
                                else:
                                    _lbp_sorted = sorted(_lbp_rec, key=lambda x: x.get('mean_anomaly_score', 0), reverse=True)
                                    _pl_lbp_n = st.number_input("Show top N", min_value=1, max_value=len(_lbp_rec), value=min(10, len(_lbp_rec)), step=1, key="pl_lbp_n_show")
                                    st.caption(f"Ranked {len(_lbp_rec)} record(s) by anomaly score (highest first)")
                                    for i, record in enumerate(_lbp_sorted[:_pl_lbp_n], 1):
                                        _rc, _ri, _cc = st.columns([0.05, 0.2, 0.75])
                                        with _rc:
                                            st.write(f"**#{i}**")
                                        with _ri:
                                            _prev = cv2.imread(record['image_path']) if os.path.exists(record['image_path']) else None
                                            if _prev is not None:
                                                _prev, _worst_kp = _draw_lbp_anomaly_kpt_on_img(_prev, record['image_path'], _lbl_map, _kpt_fmt, record, patch_size=config["lbp_patch_size"])
                                                st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                                        with _cc:
                                            _is_anom = record.get('is_anomaly', False)
                                            _anom_label = " 🔴 ANOMALY" if _is_anom else ""
                                            st.write(f"**{os.path.basename(record['image_path'])}** (Person {record['person_index']}){_anom_label}")
                                            st.write(f"Mean Score: {record.get('mean_anomaly_score', float('nan')):.4f} | Group: {record['group_id']}")
                                            _pl_sc = np.asarray(record.get('keypoint_anomaly_scores', []), dtype=float)
                                            if _pl_sc.size > 0 and not np.all(np.isnan(_pl_sc)):
                                                _pl_wk = int(np.nanargmax(_pl_sc))
                                                st.write(f"Worst keypoint: **#{_pl_wk}** (score {float(np.nanmax(_pl_sc)):.4f})")

                    # Segmentation
                    if "segmentation" in pipeline_results:
                        with col2:
                            if st.button("Segmentation Scores", use_container_width=True):
                                st.session_state.show_seg_scores = not st.session_state.get("show_seg_scores", False)
                        if st.session_state.get("show_seg_scores", False):
                            with st.expander("Segmentation - Keypoints Outside Masks", expanded=True):
                                st.write(f"**Total keypoints outside masks**: {pipeline_results['segmentation']['keypoints_outside_masks']}")
                                st.write(f"**Images evaluated**: {pipeline_results['segmentation']['evaluated_images']}")
                                _outside_kpts = pipeline_results['segmentation'].get('outside_keypoints', [])
                                # Group by image, sort by count descending
                                _pl_img_map = {}
                                for _kp in _outside_kpts:
                                    _ip = _kp.get('image_path', 'N/A')
                                    _pl_img_map.setdefault(_ip, []).append(_kp)
                                _pl_sorted = sorted(_pl_img_map.items(), key=lambda x: len(x[1]), reverse=True)
                                _pl_seg_n = st.number_input("Show top N", min_value=1, max_value=max(1, len(_pl_sorted)), value=max(1, min(20, len(_pl_sorted))), step=1, key="pl_seg_n_show")
                                for i, (_ip, _kps) in enumerate(_pl_sorted[:_pl_seg_n], 1):
                                    _rc, _ri, _cc = st.columns([0.05, 0.2, 0.75])
                                    with _rc:
                                        st.write(f"**#{i}**")
                                    with _ri:
                                        _prev = cv2.imread(_ip) if os.path.exists(_ip) else None
                                        if _prev is not None:
                                            _pl_mbp = pipeline_results['segmentation'].get('mask_by_path', {})
                                            _prev = _overlay_mask_on_img(_prev, _pl_mbp.get(os.path.abspath(_ip)))
                                            _prev = _draw_keypoints_on_img(_prev, _ip, _lbl_map, _kpt_fmt, visibility_threshold=2.0)
                                            _prev = _draw_outside_keypoints_on_img(_prev, _ip, _lbl_map, _kpt_fmt, _kps)
                                            st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                                    with _cc:
                                        st.write(f"**{os.path.basename(_ip)}** — {len(_kps)} keypoint(s) outside")
                                        _kp_ids = [f"{_k.get('keypoint_index','?')}(p{_k.get('person_index','?')})" for _k in _kps]
                                        st.write(f"Keypoints (red): {', '.join(_kp_ids)}")

                    # Distance Anomalies
                    if "distance" in pipeline_results:
                        with col3:
                            if st.button("Distance Anomalies", use_container_width=True):
                                st.session_state.show_dist_scores = not st.session_state.get("show_dist_scores", False)
                        if st.session_state.get("show_dist_scores", False):
                            with st.expander("Distance Anomalies - Error Scores", expanded=True):
                                _dist_anom = pipeline_results["distance"].get("anomalies", [])
                                _pl_dist_n = st.number_input("Show top N", min_value=1, max_value=max(1, len(_dist_anom)), value=max(1, min(10, len(_dist_anom))), step=1, key="pl_dist_n_show")
                                for position, item in enumerate(sorted(_dist_anom, key=lambda x: x['error'], reverse=True)[:_pl_dist_n], 1):
                                    _rc, _ri, _cc = st.columns([0.05, 0.2, 0.75])
                                    with _rc:
                                        st.write(f"**#{position}**")
                                    with _ri:
                                        _prev = cv2.imread(item['image_path']) if os.path.exists(item['image_path']) else None
                                        if _prev is not None:
                                            _prev, _wconn, _werr = _draw_worst_distance_on_img(
                                                _prev, item['image_path'], _lbl_map, _kpt_fmt, item
                                            )
                                            st.image(cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB), width='stretch')
                                    with _cc:
                                        st.write(f"**{os.path.basename(item['image_path'])}**")
                                        st.write(f"Mean Error: `{item['error']:.4f}`")
                                        if _wconn is not None:
                                            st.write(f"Worst distance: kp **{_wconn[0]}** → **{_wconn[1]}** (err `{_werr:.4f}`)")
        
        except Exception as e:
            st.error(f"Validation error: {e}")
            import traceback
            st.text(traceback.format_exc())
        
        st.divider()
        
        col_back, col_restart = st.columns([0.2, 0.2])
        
        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.pipeline_results = None
                st.session_state.pipeline_sub_stage = 0
                if "dup_pairs" in st.session_state:
                    del st.session_state["dup_pairs"]
                st.session_state.deleted_basenames = set()
                st.session_state.validation_stage = 1
                st.rerun()
        
        with col_restart:
            if st.button("New Validation", use_container_width=True, type="primary"):
                st.session_state.pipeline_results = None
                st.session_state.pipeline_sub_stage = 0
                if "dup_pairs" in st.session_state:
                    del st.session_state["dup_pairs"]
                st.session_state.deleted_basenames = set()
                st.session_state.validation_stage = 0
                st.rerun()
