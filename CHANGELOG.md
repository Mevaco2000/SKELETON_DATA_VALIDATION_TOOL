# Changelog

## [0.2.0] - 2026-03-04

### Added

#### Validation Module - Image and Dataset Classes
- **YPImageValidation Class** — Per-image LBP feature extraction
  - `compute_lbp_histograms(patch_size)` — LBP histograms for all keypoints
  - `compute_lbp_decimals(patch_size)` — Binary decimal representations
  - `compute_lbp_histogram_single(x, y, patch_size)` — Single keypoint histogram
  - `compute_lbp_decimal_single(x, y, patch_size)` — Single keypoint decimal
  - `sequential_distances(visibility_threshold)` — Distance between keypoints
  - `visualize(patch_size)` — Visualize with LBP patches
  - Static: `load_keypoints()`, `draw_keypoints_on_image()`, `compute_sequential_distances()`

- **YPSetValidation Class** — Dataset-level feature extraction
  - `from_yolo_dataset(dataset, patch_size)` — Create from YOLOPoseDataset
  - `get_all_lbp_histograms()` — Extract all LBP histograms
  - `get_all_lbp_decimals()` — Extract all binary decimals
  - `create_dataframe()` — Export to pandas DataFrame

#### Dataset Module - Core Classes
- **YOLOPoseImage** — Single imagewith annotations
  - Load image and keypoints
  - Access normalized coordinates
  - To/from dict conversion

- **YOLOPoseDataset** — Full dataset management
  - Load from data.yaml, train.txt, or args.yaml
  - Iterator interface
  - Get statistics and paths
  - Efficient memory usage

### Changed

#### Code Organization
- **validation/**: Now includes
  - `image_validation.py` — YPImageValidation class
  - `set_validation.py` — YPSetValidation class
  - `helpers.py` — Utility functions
  - `analysis.py` — Group analysis

- **datasets/**: Now includes
  - `yolo_pose_dataset.py` — YOLOPoseDataset and YOLOPoseImage classes
  - `operations.py` — Dataset operations

### Deprecated
- References to non-existent modules (duplicates.py, evaluation.py)
- Stand-alone LBP computation functions (now in YPImageValidation)

### Fixed
- Consolidated class-based interface for cleaner API
- Removed redundant function definitions
- Improved code organization and maintainability

---

## [0.1.0] - 2026-02-26

### Added

#### Validation Module
- **helpers.py**: Keypoint utilities
  - `load_keypoints()` — Load labels from YOLO files
  - `draw_keypoints()` — Visualize keypoints

- **analysis.py**: Group analysis and reporting
  - `generate_group_visualizations()` — PNG visualizations
  - `save_groups_analysis()` — JSON export
  - `analyze_hidden_keypoints()` — Visibility analysis
  - `export_groups_analysis_to_excel()` — Excel reports

#### Datasets Module
- **operations.py**: Dataset manipulation
  - `flatten_cvat_yolo_pose()` — Flatten CVAT structures
  - `merge_yolo_pose_datasets()` — Merge multiple datasets
  - `split_yolo_pose_dataset()` — Train/validation split
  - `convert_yolo_pose_to_cvat()` — Convert to CVAT format
  - `extract_image_subset()` — Extract by index range

#### Configuration
- **config.py**: Centralized constants
  - Model registry
  - Image format settings
  - Default thresholds

#### Package Structure
- Professional module organization
- Clean `__init__.py` exports
- Full type hints
- NumPy-style docstrings

### Documentation
- README.md — Full API documentation
- QUICK_START.md — 5-minute setup guide
- CHANGELOG.md — Version tracking

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
