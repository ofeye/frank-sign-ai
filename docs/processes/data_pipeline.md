# Process: Data Pipeline Development

## Status: 🟢 Active
## Last Updated: 2026-01-14
## Owner: AI Agent + Human

---

## Current Objectives

- [x] CVAT annotation parser
- [x] Geometric feature extractor
- [x] Clinical data loader
- [ ] Image preprocessing pipeline
- [ ] PyTorch Dataset class
- [ ] Data versioning (DVC)

---

## Progress Log

### 2026-01-14
**Completed:**
- Added CLI skeleton (`franksign-parse`, train/eval placeholders) aligned with pyproject entrypoints
- Scaffolded torch `FrankSignDataset` and lightweight preprocess helper
- Added Pandera-based `scripts/validate_data.py` for clinical CSV checks

**Blockers:**
- Full preprocessing (ruler detection) and Dataset transforms pending

**Next Steps:**
- Integrate production-ready preprocessing/augmentation
- Add Dataset-based DataLoader and training loop
- Expand validation schema for new clinical columns

### 2026-01-13
**Completed:**
- Created `clinical_loader.py` with Turkish format parsing
- Created `clinical_data_schema.md` documentation
- Verified loading of 204 sample patients
- Created test suite for clinical loader

**Blockers:**
- None currently

**Next Steps:**
- Image preprocessing (ruler detection)
- Create unified Dataset class

### 2026-01-12
**Completed:**
- Created `cvat_parser.py` - parses 14 label types
- Created `geometric_features.py` - curvature, tortuosity
- Verified 121 images, 93 with Frank Sign

---

## Dependencies

**Depends on:**
- None (foundation process)

**Blocking:**
- Model Development (needs Dataset class)
- Clinical Integration (needs complete data pipeline)

---

## Key Decisions Made

1. **Turkish decimal handling**: Custom parsing functions in clinical_loader
2. **Sample data approach**: Marked clearly, code designed for easy transition
3. **Virtual environment**: Using `.venv/` instead of conda

---

## Open Questions

1. ❓ Should ruler detection be automated or manual?
2. ❓ Augmentation strategy for ear images (no horizontal flip - asymmetric)
3. ❓ How to handle images without matching clinical data?

---

## Files Involved

| File | Purpose |
|------|---------|
| `src/franksign/data/cvat_parser.py` | Annotation parsing |
| `src/franksign/data/geometric_features.py` | Feature extraction |
| `src/franksign/data/clinical_loader.py` | Clinical data loading |
| `src/franksign/data/dataset.py` | PyTorch Dataset (planned) |
| `scripts/parse_annotations.py` | CLI tool |
