# Changelog

## [0.2.0] - 2026-03-03

### Added

#### Validation Module - LBP Feature Extraction
- **YOLOPoseDataset Class** — Unified dataset interface
  - Supports data.yaml input (auto-detects train/val/test splits)
  - Supports train.txt and args.yaml inputs
  - Iterator interface for memory-efficient loading
  - Methods for feature extraction:
    - `get_all_lbp_histograms(patch_size)` — Compute all LBP histograms
    - `get_all_lbp_decimals(patch_size)` — Compute binary decimals
    - `iterate_with_lbp(patch_size)` — Iterate with features
    - `apply_function_to_all(func)` — Apply custom functions
    - `get_sample_with_features(idx, patch_size)` — Single sample with features
    - `create_dataframe(patch_size)` — Export to pandas DataFrame
    - `visualize_sample(idx, patch_size)` — Visualize with patches

- **LBP Core Functions** — Enhanced with dataset examples
  - `compute_lbp_for_image()` — LBP histograms for all keypoints
  - `compute_lbp_for_keypoints()` — Alias with same functionality
  - `compute_lbp_histogram()` — Single keypoint LBP
  - `compute_lbp_value()` — Basic 3x3 LBP computation
  - `compute_lbp_binary_decimal()` — Binary decimal representation
  - `patch_to_binary_decimal()` — Patch to decimal conversion

### Changed

#### Code Organization
- **Removed redundant functions** (consolidated into YOLOPoseDataset):
  - ~~`compute_lbp_batch()`~~ → Use `dataset.get_all_lbp_histograms()`
  - ~~`load_image_and_keypoints()`~~ → Use `dataset[idx]`
  - ~~`load_image_and_keypoints_from_dataset()`~~ → Use `for sample in dataset`
  - ~~`load_images_from_dataset()`~~ → Use `dataset` iteration
  - ~~`compute_lbp_for_dataset()`~~ → Use `dataset.get_all_lbp_histograms()`

- **evaluation.py** — Reduced from 1176 to 950 lines
  - Better code organization with class-based interface
  - All standalone functions retained with enhanced examples

### Fixed
- Fixed duplicate docstrings in `compute_lbp_for_image()`
- Corrected indentation in docstring examples

---

## [0.1.0] - 2026-02-26

### Added

#### Validation Module
- **helpers.py**: Keypoint loading, distance calculations, and visualization
  - `load_keypoints()` — Load keypoints from YOLO labels
  - `sequential_distances()` — Calculate keypoint-to-keypoint distances
  - `draw_keypoints()` — Draw keypoints on images

- **duplicates.py**: CLIP-based duplicate detection
  - `generate_clip_embeddings()` — Generate CLIP embeddings for images
  - `find_near_duplicates()` — Find similar images using FAISS
  - Private helpers: `_load_clip_model()`, `_compute_embeddings()`, `_build_similarity_groups()`

- **analysis.py**: Group analysis and reporting
  - `generate_group_visualizations()` — Create PNG visualizations with overlaid keypoints
  - `save_groups_analysis()` — Export groups to JSON with distance vectors
  - `analyze_hidden_keypoints()` — Detect hidden/invalid keypoints
  - `export_groups_analysis_to_excel()` — Generate Excel reports with statistics

- **evaluation.py**: YOLO model training and evaluation
  - `train_yolo_pose_model()` — Train YOLO pose models with registry support
  - `load_yolo_pose_label()` — Load labels as keypoint arrays
  - `evaluate_model_on_dataset()` — Generate detailed error reports

#### Datasets Module
- **operations.py**: Dataset manipulation utilities
  - `flatten_cvat_yolo_pose()` — Flatten nested CVAT structures
  - `merge_yolo_pose_datasets()` — Merge multiple datasets with conflict handling
  - `split_yolo_pose_dataset()` — Split into train/validation sets
  - `convert_yolo_pose_to_cvat()` — Convert to CVAT ZIP format
  - `extract_image_subset()` — Extract images by index range

#### Configuration
- **config.py**: Centralized constants
  - Model registry with predefined paths
  - Image format whitelist
  - CLIP model settings
  - Default thresholds for similarity and duplicate detection
  - Default YOLO training parameters

#### Package Structure
- Professional `__init__.py` files with clean API exports
- Full type hints on all functions
- NumPy-style docstrings for all public functions
- Module-level documentation strings

### Documentation
- **README.md**: Comprehensive documentation with 5 sections and full API reference
- **QUICK_START.md**: 5-minute setup guide with common usage patterns
- **CHANGELOG.md**: Version history and change tracking

### Type Safety
- Full type annotations on all parameters and return values
- Ready for `mypy` type checking
- IDE autocomplete support

### Error Handling
- Informative exception messages
- Input validation on all public functions
- Proper error types: `FileNotFoundError`, `IndexError`, `ValueError`, `TypeError`

### Progress Tracking
- `tqdm` progress bars on all long-running operations
- Automatic ETA calculation

### Features
- GPU acceleration support (CUDA with fallback to CPU)
- Automatic skipping of corrupted images (no errors raised)
- Path normalization across platforms
- Seed-based reproducibility

---

## Notable Design Decisions

### 1. Module Organization by Function
- **validation/**: Image quality and duplicate detection
- **datasets/**: File system operations and format conversion
- Logical separation allows independent module updates

### 2. Public vs Private Functions
Functions prefixed with `_` are private/internal:
- `_load_clip_model()` — Only used internally
- `_compute_embeddings()` — Wrapped by public `generate_clip_embeddings()`
- `_build_similarity_groups()` — Internal grouping logic

### 3. Centralized Configuration
All magic constants in `config.py`:
- Easy project-wide tuning without code changes
- Single source of truth for all defaults

### 4. Type Hints Everywhere
- Full annotations enable IDE autocomplete
- Type checker compatibility (`mypy`)
- Self-documenting code

### 5. Graceful Error Handling
- No silent failures on corrupt images
- Clear error messages for debugging
- Proper exception types for programmatic handling

---

## Known Limitations

- CLIP embeddings require significant GPU memory (~8GB VRAM)
- FAISS similarity search scales O(n*k) with dataset size
- Excel export limited by `openpyxl` row/column limits (~1 million rows)
- YOLO training requires compatible CUDA installation

---

## Future Roadmap

### v0.2.0 (Planned)
- [ ] Async/await support for long-running operations
- [ ] Batch processing for very large datasets
- [ ] Caching of CLIP embeddings to disk
- [ ] Web API layer (FastAPI)
- [ ] Docker containerization

### v0.3.0 (Planned)
- [ ] Multi-GPU support
- [ ] Advanced statistics (percentile analysis, outlier detection)
- [ ] Annotation format converters (COCO, Pascal VOC)
- [ ] Model ensemble evaluation

### v1.0.0 (Planned)
- [ ] Stable API freeze
- [ ] Full test suite with >90% coverage
- [ ] Performance benchmarks
- [ ] Community contributions

---

## Credits

**Developer:** Data Validation Team  
**Last Updated:** February 26, 2026  
**License:** © 2026 Data Validation Team
