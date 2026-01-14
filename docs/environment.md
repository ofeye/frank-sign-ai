# Development Environment Setup

> This document tracks the project's Python environment and dependencies.

## Virtual Environment

**Location**: `.venv/` (in project root)  
**Python Version**: 3.11  
**Created**: 2026-01-13

### Activation
```bash
cd "/Applications/Codes/84. AI_FrankSign"
source .venv/bin/activate
```

### Deactivation
```bash
deactivate
```

---

## Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.4.1 | Numerical operations, array processing |
| pandas | 2.3.3 | Clinical data loading and manipulation |
| lxml | 6.0.2 | CVAT XML annotation parsing |
| pyyaml | 6.0.3 | Configuration file parsing |
| pytest | 9.0.2 | Unit testing |
| pandera | _planned_ | DataFrame schema validation (not yet installed in venv) |

> Forward-looking dependencies (torch, torchvision, segmentation-models-pytorch, etc.) remain specified in `pyproject.toml` but are not installed in the current lightweight venv.

### Installation Log

```
2026-01-13: Initial environment setup
- pip upgraded to 25.3
- Core packages installed: pandas, numpy, lxml, pyyaml, pytest
```

---

## Adding New Packages

When adding new packages:

1. **Activate environment**: `source .venv/bin/activate`
2. **Install package**: `pip install <package_name>`
3. **Update this doc**: Add to packages table above
4. **Update requirements**: `pip freeze > requirements.txt`

### Future Packages (Planned)

| Package | Version | Purpose | Phase |
|---------|---------|---------|-------|
| torch | 2.x | Deep learning | Phase 3 |
| torchvision | 0.x | Image transforms | Phase 3 |
| opencv-python | 4.x | Image processing | Phase 2 |
| scikit-image | 0.x | Advanced image ops | Phase 2 |
| albumentations | 1.x | Augmentation | Phase 2 |
| segmentation-models-pytorch | 0.3+ | Model architectures | Phase 3 |
| captum | 0.6+ | XAI (Grad-CAM) | Phase 3 |
| mlflow | 2.x | Experiment tracking | Phase 4 |

---

## Environment Issues & Solutions

### Issue 1: Conda NumPy Conflict (2026-01-12)
**Problem**: libgfortran.5.dylib not found in conda environment  
**Solution**: Created fresh venv instead of using conda

### Issue 2: (Reserved for future issues)

---

## Verification

After setting up, verify with:
```bash
source .venv/bin/activate
python -c "import pandas; import numpy; import lxml; print('✅ All packages work!')"
pytest tests/ -v  # Run all tests
```
