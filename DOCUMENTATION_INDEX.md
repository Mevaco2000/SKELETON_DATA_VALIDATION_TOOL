# Documentation Index

Complete reference for all documentation files in the project.

---

## 📚 Documentation Files

| File | Purpose | Read Time | Best For |
|------|---------|-----------|----------|
| **README.md** | Comprehensive API reference | 15 min | Full documentation, all features |
| **QUICK_START.md** | 5-minute setup & common tasks | 5 min | Getting started, copying examples |
| **API_CHEATSHEET.md** | Quick function lookup | 3 min | Fast reference, copy-paste |
| **PROJECT_STRUCTURE.md** | Architecture & organization | 8 min | Understanding code structure |
| **CHANGELOG.md** | Version history & features | 5 min | Release notes, what's new |

---

## 🎯 Quick Navigation

### I want to...

#### **Get started quickly**
→ Read [QUICK_START.md](QUICK_START.md) (5 min)
```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPImageValidation
```

#### **Look up a function or class**
→ Check [API_CHEATSHEET.md](API_CHEATSHEET.md) (3 min)
- All class signatures with examples
- Quick copy-paste code

#### **Learn the full API**
→ Read [README.md](README.md) (15 min)
- Complete documentation
- Detailed examples
- Configuration options

#### **Understand the structure**
→ Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (8 min)
- Module organization
- File layout
- Design patterns

#### **See what's new**
→ Check [CHANGELOG.md](CHANGELOG.md) (5 min)
- Version history
- New features
- What changed

---

## 📖 Reading Paths by Role

### 👨‍💻 Developer (First Time)
1. [QUICK_START.md](QUICK_START.md) — Get it working (5 min)
2. [API_CHEATSHEET.md](API_CHEATSHEET.md) — Learn the API (3 min)
3. [README.md](README.md) — Deep dive (15 min)

**Total: ~23 minutes**

### 📊 Data Scientist / Analyst
1. [QUICK_START.md](QUICK_START.md) — Setup & examples (5 min)
2. [API_CHEATSHEET.md](API_CHEATSHEET.md) — Bookmark for reference (3 min)
3. [README.md](README.md) — Detailed examples (15 min)

**Total: ~23 minutes + keep API_CHEATSHEET bookmarked**

### 🔧 Maintainer / Contributor
1. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Architecture (8 min)
2. [README.md](README.md) — Full API (15 min)
3. [CHANGELOG.md](CHANGELOG.md) — Version history (5 min)

**Total: ~28 minutes + code review**

### ⚡ Power User (Familiar with Python)
1. [API_CHEATSHEET.md](API_CHEATSHEET.md) — Skim signatures (1 min)
2. `help(ClassName)` in Python — As needed
3. [README.md](README.md) — Reference only

---

## 🔑 Key Concepts

### Main Classes

**YOLOPoseDataset** — Load YOLO datasets
```python
from utils.datasets import YOLOPoseDataset
dataset = YOLOPoseDataset("data.yaml")
```

**YOLOPoseImage** — Single image with annotations
```python
from utils.datasets import YOLOPoseImage
img = YOLOPoseImage("image.jpg", "image.txt")
```

**YPImageValidation** — Per-image LBP features
```python
from utils.validation import YPImageValidation
validator = YPImageValidation(image, keypoints)
```

**YPSetValidation** — Dataset-level features
```python
from utils.validation import YPSetValidation
validator = YPSetValidation.from_yolo_dataset(dataset)
```

### Main Operations

**Dataset Operations**
- `split_yolo_pose_dataset()` — Split train/validation
- `merge_yolo_pose_datasets()` — Merge datasets
- `flatten_cvat_yolo_pose()` — Convert CVAT format
- `extract_image_subset()` — Extract images by range

**Analysis Functions**
- `generate_group_visualizations()` — Visualize groups
- `save_groups_analysis()` — JSON report
- `export_groups_analysis_to_excel()` — Excel report

**Utilities**
- `YPImageValidation.load_keypoints()` — Load from label file
- `YPImageValidation.draw_keypoints_on_image()` — Draw on image

---

## 💡 Common Use Cases

### Use Case 1: Load and analyze dataset
```python
from utils.datasets import YOLOPoseDataset
from utils.validation import YPSetValidation

dataset = YOLOPoseDataset("data.yaml")
validator = YPSetValidation.from_yolo_dataset(dataset, patch_size=5)
features = validator.get_all_lbp_histograms()
```
→ See: [QUICK_START.md - Task 3](QUICK_START.md), [API_CHEATSHEET.md](API_CHEATSHEET.md)

### Use Case 2: Process single image
```python
from utils.datasets import YOLOPoseImage
from utils.validation import YPImageValidation

img = YOLOPoseImage("image.jpg", "image.txt")
validator = YPImageValidation(img.image, img.keypoints)
features = validator.compute_lbp_histograms(patch_size=5)
```
→ See: [QUICK_START.md - Task 2](QUICK_START.md), [API_CHEATSHEET.md](API_CHEATSHEET.md)

### Use Case 3: Dataset operations
```python
from utils.datasets import split_yolo_pose_dataset
split_yolo_pose_dataset("data", "data_split", val_ratio=0.2)
```
→ See: [QUICK_START.md - Tasks 4-6](QUICK_START.md), [API_CHEATSHEET.md - Dataset Operations](API_CHEATSHEET.md)

---

## ❓ FAQ

**Q: How do I load a dataset?**  
A: Use `YOLOPoseDataset("data.yaml")` — See [QUICK_START.md - Task 1](QUICK_START.md)

**Q: How do I get LBP features?**  
A: Use `YPImageValidation` for images or `YPSetValidation` for datasets — See [QUICK_START.md - Tasks 2-3](QUICK_START.md)

**Q: Where's the complete API?**  
A: Check [API_CHEATSHEET.md](API_CHEATSHEET.md) for all functions and classes

**Q: How does the project structure work?**  
A: Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**Total: 1 minute + lookup**

---

## 🗂️ Documentation Structure

```
Project Documentation
│
├── User Guides
│   ├── QUICK_START.md .................... New users, quick intro
│   └── README.md ......................... Full comprehensive guide
│
├── Reference Materials
│   ├── API_CHEATSHEET.md ................ Function signatures & quick lookup
│   ├── PROJECT_STRUCTURE.md ............ Architecture & organization
│   └── pyproject.toml ................... Package metadata
│
├── Maintenance Docs
│   ├── CHANGELOG.md ..................... Version history & roadmap
│   ├── .gitignore ....................... Version control patterns
│   └── setup.py ......................... Legacy packaging
│
└── Configuration
    └── requirements.txt ................. Dependencies
```

---

## 📝 Content Summary

### README.md
- Package overview
- Installation instructions
- 5 core modules explained
- Configuration options
- 5 detailed usage examples
- Complete API reference
- Troubleshooting guide

### QUICK_START.md
- 5-minute setup
- 6 common tasks with code
- Directory structure
- File examples
- Performance tips
- Troubleshooting table

### API_CHEATSHEET.md
- All function signatures
- Quick copy-paste examples
- Common patterns (3 examples)
- Error handling
- Performance tips
- Function reference table

### PROJECT_STRUCTURE.md
- Directory layout
- Module descriptions
- Organization principles
- File patterns
- Data flow diagrams
- Import best practices
- Design philosophy

### CHANGELOG.md
- v0.2.0 release notes (LBP Features, YOLOPoseDataset improvements)
- v0.1.0 release notes
- Feature list by module
- Design decisions explained
- Known limitations
- Future roadmap (v0.3, v1.0)

---

## 🔍 Search Index

### By Topic

**Duplicate Detection:**
- QUICK_START.md → "Find Duplicate Images"
- README.md → "Validation Module" → "duplicates.py"
- API_CHEATSHEET.md → "CLIP Embeddings & Duplicates"

**Dataset Operations:**
- QUICK_START.md → "Split Dataset", "Extract Image Subset"
- README.md → "Datasets Module"
- API_CHEATSHEET.md → "Datasets Module"

**YOLO Training:**
- QUICK_START.md → "Train YOLO Model"
- README.md → "Evaluation Module"
- API_CHEATSHEET.md → "YOLO Model Training & Evaluation"

**LBP Feature Extraction:**
- QUICK_START.md → "Extract LBP Features from Dataset"
- README.md → "LBP Feature Extraction"
- API_CHEATSHEET.md → "LBP Features" section
- YOLOPoseDataset class with `get_all_lbp_histograms()`, `get_all_lbp_decimals()`

**Keypoint Analysis:**
- README.md → "Keypoint Utilities"
- API_CHEATSHEET.md → "Keypoint Utilities"
- QUICK_START.md → "Load Keypoints"

**Configuration:**
- README.md → "Configuration" section
- PROJECT_STRUCTURE.md → "Centralized Configuration"
- QUICK_START.md → "Tips" section

**Installation:**
- QUICK_START.md → "5-Minute Setup"
- README.md → "Installation"
- requirements.txt and setup.py

---

## 💡 Tips

✅ **Bookmark these:**
- `QUICK_START.md` for your first run
- `API_CHEATSHEET.md` for quick lookups
- `README.md` for detailed reference

✅ **Refer to examples in:**
- QUICK_START.md — Common real-world tasks
- README.md → "Usage Examples" — Complex scenarios
- API_CHEATSHEET.md → "Common Patterns" — Best practices

✅ **For debugging:**
1. Check error message in console
2. Look in [README.md](README.md) → "Troubleshooting" section
3. Check [API_CHEATSHEET.md](API_CHEATSHEET.md) → "Error Handling"
4. Run `help(function_name)` in Python

---

## 🚀 Next Steps

1. **Start here:** [QUICK_START.md](QUICK_START.md)
2. **Reference:** [API_CHEATSHEET.md](API_CHEATSHEET.md)
3. **Deep dive:** [README.md](README.md)
4. **Extend:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 📞 Support Resources

| Resource | When to Use |
|----------|------------|
| This Index | Lost or need direction |
| QUICK_START.md | First time setup |
| API_CHEATSHEET.md | Quick function lookup |
| README.md | Detailed explanation |
| PROJECT_STRUCTURE.md | Understanding architecture |
| `help(function_name)` | In-code documentation |
| CHANGELOG.md | Version tracking |

---

## Version Info

- **Package Version:** 0.2.0
- **Last Updated:** March 3, 2026
- **Documentation Version:** 1.1

---

**Happy coding! 🎉**

If you can't find what you need:
1. Check the index above
2. Browse the appropriate file
3. Use Python's `help()` function
4. Search within `README.md` for keywords
