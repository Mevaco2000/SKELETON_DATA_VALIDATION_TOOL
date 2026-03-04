import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from typing import Optional
import sys

# Add parent directory to path for imports
gui_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(gui_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from utils.datasets.yolo_pose_dataset import YOLOPoseDataset, YOLOPoseImage
    from utils.validation.image_validation import YPImageValidation
except ImportError as e:
    st.error(f"❌ Import error: {str(e)}")
    st.stop()


st.set_page_config(page_title="YOLO Pose Dataset Validation", layout="wide")
st.title("YOLO Pose Dataset Validation")
st.write("This is a Streamlit app for validating YOLO Pose datasets. Use the sidebar to navigate through different validation tools and analyses.")


def convert_bgr_to_rgb(image_bgr):
    """Convert BGR image (from cv2) to RGB for displaying with Streamlit."""
    if image_bgr is not None and len(image_bgr.shape) == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr


def draw_keypoints_on_image(image, keypoints, visibility_threshold=0.5, color=(0, 255, 0), radius=4):
    """Draw keypoints as circles on image."""
    image_copy = image.copy() if isinstance(image, np.ndarray) else image
    h, w = image_copy.shape[:2]
    
    # Handle visibility scores in 3rd column
    if keypoints.shape[1] >= 3:
        visibility = keypoints[:, 2]
        visible_mask = visibility >= visibility_threshold
    else:
        visible_mask = np.ones(len(keypoints), dtype=bool)
    
    for i, (x, y) in enumerate(keypoints[:, :2]):
        if visible_mask[i]:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(image_copy, (px, py), radius, color, -1)
            # Draw keypoint number
            cv2.putText(image_copy, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image_copy


# Sidebar configuration
with st.sidebar:
    st.header("Dataset Configuration")
    
    # Initialize dataset session state
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    
    # Dataset loading options
    st.subheader("📁 Load Dataset")
    
    # Option 1: Upload file
    uploaded_file = st.file_uploader(
        "Or upload data.yaml / train.txt",
        type=["yaml", "yml", "txt"],
        key="file_uploader"
    )
    
    dataset_input = ""
    
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            dataset_input = tmp_file.name
            st.success(f"✅ File: {uploaded_file.name}")
    
    st.divider()
    
    # Load dataset button
    if st.button("🚀 Load Dataset", key="load_btn", use_container_width=True):
        if dataset_input:
            dataset_input = os.path.abspath(dataset_input)
            if os.path.exists(dataset_input):
                try:
                    st.session_state.dataset = YOLOPoseDataset(dataset_input)
                    st.session_state.current_index = 0
                    st.success(f"✅ Loaded! {len(st.session_state.dataset)} images found")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ File not found. Check the path and try again.")
        else:
            st.error("❌ Please upload or enter a dataset path.")
    
    st.divider()
    
    # Display dataset info
    if st.session_state.dataset is not None:
        st.subheader("Dataset Info")
        st.metric("Total Images", len(st.session_state.dataset))
        st.metric("Dataset Dir", st.session_state.dataset.dataset_dir)
        st.metric("Labels Dir", st.session_state.dataset.labels_dir)
    
    st.divider()
    
    # Visualization options
    st.subheader("Display Options")
    visibility_threshold = st.slider("Visibility Threshold", 0.0, 1.0, 0.5, 0.1)
    keypoint_color = st.color_picker("Keypoint Color", "#00FF00")
    keypoint_size = st.slider("Keypoint Size", 1, 20, 4)
    
    # Convert hex color to BGR for cv2
    hex_color = keypoint_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert RGB to BGR


# Main content area
if st.session_state.dataset is not None:
    dataset = st.session_state.dataset
    
    # Image navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", key="prev_btn"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    
    with col2:
        st.session_state.current_index = st.slider(
            "Select Image",
            0,
            len(dataset) - 1,
            st.session_state.current_index,
            key="image_slider"
        )
    
    with col3:
        if st.button("Next ➡️", key="next_btn"):
            st.session_state.current_index = min(len(dataset) - 1, st.session_state.current_index + 1)
    
    # Load and display current sample
    try:
        idx = st.session_state.current_index
        sample = dataset[idx]
        
        image = sample['image']
        keypoints = sample['keypoints']
        image_path = sample['image_path']
        
        # Convert BGR to RGB for display
        image_rgb = convert_bgr_to_rgb(image)
        
        # Draw keypoints on image
        image_with_kpts = draw_keypoints_on_image(
            image_rgb, 
            keypoints, 
            visibility_threshold=visibility_threshold,
            color=bgr_color,
            radius=keypoint_size
        )
        
        # Display image with layout
        st.subheader(f"Image {idx + 1} / {len(dataset)}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image_with_kpts, caption=f"Image with Keypoints", use_column_width=True)
        
        with col2:
            st.write("**Image Info:**")
            st.write(f"Path: {image_path}")
            st.write(f"Shape: {image.shape}")
            st.write(f"**Keypoints Info:**")
            st.write(f"Total keypoints: {len(keypoints)}")
            
            # Count visible keypoints
            if keypoints.shape[1] >= 3:
                visible_count = np.sum(keypoints[:, 2] >= visibility_threshold)
                st.write(f"Visible: {visible_count} / {len(keypoints)}")
            
            # Get sequential distances
            try:
                validator = YPImageValidation(image, keypoints)
                distances = validator.compute_sequential_distances(visibility_threshold=visibility_threshold)
                if len(distances) > 0:
                    st.write(f"**Sequential Distances:**")
                    st.write(f"Mean: {distances.mean():.4f}")
                    st.write(f"Min: {distances.min():.4f}")
                    st.write(f"Max: {distances.max():.4f}")
            except Exception as e:
                st.warning(f"Could not compute distances: {str(e)}")
        
        # Keypoints table
        with st.expander("View Keypoints Data"):
            if keypoints.shape[1] == 3:
                df = pd.DataFrame(
                    keypoints,
                    columns=['X (norm)', 'Y (norm)', 'Visibility']
                )
            else:
                df = pd.DataFrame(
                    keypoints,
                    columns=['X (norm)', 'Y (norm)']
                )
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")

else:
    # Show placeholder if no dataset loaded
    st.info("👈 Load a dataset using the sidebar to view images")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Supported Formats")
        st.write("""
        - **data.yaml** - YOLO dataset config
        - **train.txt** - Image list file
        """)
    
    with col2:
        st.subheader("Features")
        st.write("""
        - 🖼️ Image browser
        - 🔵 Keypoint visualization
        - 📊 Keypoint statistics
        - 📈 Sequential distances
        """)
