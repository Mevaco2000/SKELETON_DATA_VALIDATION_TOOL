# Documentation Index

Complete reference for all documentation files in the project.

---

## 📚 Documentation Files

| File | Purpose | Read Time | Audience |
|------|---------|-----------|----------|
| **README.md** | Full API reference & comprehensive guide | 10 min | Developers, Data Scientists |
| **QUICK_START.md** | 5-minute setup & common tasks with examples | 5 min | New users, Quick reference |
| **API_CHEATSHEET.md** | Quick function reference & patterns | 3 min | Speed lookup, troubleshooting |
| **PROJECT_STRUCTURE.md** | Package organization & architecture | 8 min | Maintainers, Contributors |
| **CHANGELOG.md** | Version history & roadmap | 5 min | Release notes, planning |
| **pyproject.toml** | Python packaging metadata | - | Package managers |
| **setup.py** | Legacy Python setup | - | pip compatibility |
| **requirements.txt** | Pip dependencies | - | Virtual environment setup |
| **.gitignore** | Git ignore patterns | - | Version control |

---

## 🎯 Quick Navigation

### I want to...

#### **Get started quickly**
→ Read [QUICK_START.md](QUICK_START.md) (5 min)
```python
from utils import load_keypoints, train_yolo_pose_model
```

#### **Look up a function**
→ Check [API_CHEATSHEET.md](API_CHEATSHEET.md) (3 min)
```python
# Jump to function signatures and examples
```

#### **Learn the full API**
→ Read [README.md](README.md) (10 min)
- Complete function documentation
- Usage examples
- Configuration options
- Troubleshooting

#### **Understand the project structure**
→ Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (8 min)
- Module organization
- Design principles
- Import patterns
- Data flow diagrams

#### **See what's new**
→ Check [CHANGELOG.md](CHANGELOG.md)
- Version history
- New features
- Known limitations
- Roadmap

#### **Install dependencies**
→ Use [requirements.txt](requirements.txt)
```bash
pip install -r requirements.txt
```

#### **Extend the package**
→ Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) → "Future Extensibility" section

---

## 📖 Reading Paths by Role

### 👨‍💻 Developer (First Time)
1. [QUICK_START.md](QUICK_START.md) — Get it working (5 min)
2. [API_CHEATSHEET.md](API_CHEATSHEET.md) — Learn the functions (3 min)
3. [README.md](README.md) — Deep dive (10 min)
4. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Understand architecture (8 min)

**Total: ~26 minutes**

### 🔧 Maintainer / Contributor
1. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Architecture overview (8 min)
2. [README.md](README.md) — Full API documentation (10 min)
3. [CHANGELOG.md](CHANGELOG.md) — Planned features (5 min)
4. Code review in `utils/` package

**Total: ~23 minutes + code review**

### 📊 Data Scientist / Analyst
1. [QUICK_START.md](QUICK_START.md) — Setup and common tasks (5 min)
2. [README.md](README.md) — Full API with examples (10 min)
3. [API_CHEATSHEET.md](API_CHEATSHEET.md) — Quick reference bookmark

**Total: ~15 minutes + bookmarks**

### ⚡ Power User (Knows Python)
1. [API_CHEATSHEET.md](API_CHEATSHEET.md) — skim signatures (1 min)
2. `help(function_name)` in Python (as needed)
3. [README.md](README.md) — Reference only (as needed)

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
