# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with modular Python package layout
- README.md with project overview and quick start guide
- AGENTS.md for AI coding assistant guidelines (continuous monitoring, sample data awareness)
- CHANGELOG.md for version tracking
- ROADMAP.md for project phase documentation
- Directory structure: src/, data/, models/, experiments/, configs/, docs/, scripts/
- Configuration system with YAML support (`configs/default.yaml`)
- CVAT annotation XML parser (`cvat_parser.py`) - parses 14 label types
- Geometric feature extractor (`geometric_features.py`) - curvature, tortuosity, area
- Clinical data loader (`clinical_loader.py`) - handles Turkish decimal format
- Clinical data schema documentation (`docs/clinical_data_schema.md`)
- Environment documentation (`docs/environment.md`)
- Data management guide (`docs/data_management.md`) - sample vs production data
- Parallel process tracking (`docs/processes/`) for multi-stream work
- Unit tests for parser and clinical loader
- CLI script for annotation parsing (`scripts/parse_annotations.py`)
- Virtual environment (.venv) with Python 3.11, numpy 2.4.1, pandas 2.3.3
- **Package-wide logging configuration** (`src/franksign/__init__.py`) (2026-01-13)
- **Pre-commit hooks** (`.pre-commit-config.yaml`) - Black, Ruff, isort (2026-01-13)
- **Lead Developer Audit Report** (`docs/audit_report_draft.md`) (2026-01-13)
- **Documentation Sync Protocol** in AGENTS.md - MUST rules for doc updates (2026-01-13)
- **CLI skeleton** (`src/franksign/cli/`) with parse/train/eval entrypoints (2026-01-14)
- **Dataset/preprocess scaffolds** (`src/franksign/data/dataset.py`, `preprocess.py`) (2026-01-14)
- **Validation utilities** (`src/franksign/data/validation.py`) incl. Pandera schema and CVAT checks (2026-01-14)
- **Clinical validation script** (`scripts/validate_data.py`) using Pandera (2026-01-14)
- **task.md** tracker (2026-01-14)

### Changed
- ROADMAP.md Phase 3: Added MAEF-Net and Mamba-UNet to model experimental design (2026-01-13)
- README quick start now points to package entrypoints and validation script (2026-01-14)
- Data pipeline process log updated with dataset/validation scaffolds (2026-01-14)
- pyproject.toml dependencies include Pandera for schema validation (2026-01-14)
- `validate_data.py` can also check CVAT annotations structurally (2026-01-14)

### Fixed
- CVAT parser: _parse_point now handles semicolon-separated multi-point coordinates

### Verified Data
- 121 images in CVAT annotations (93 with Frank Sign line)
- 204 patients in clinical CSV (91 with Frank Sign left)

### Planned
- PyTorch Dataset implementation (images + clinical data)
- Image preprocessing pipeline (ruler detection, augmentation)
- Attention U-Net, MAEF-Net, Mamba-UNet models

---

## Version History Format

Each version entry should include:
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Features that will be removed
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

---

## [0.1.0] - 2026-01-12

### Added
- Project initialization
- TÜBİTAK 1005 proposal documentation (`Project_Main.md`)
- CVAT pilot annotations for 121 images (`annotations.xml`)
- Annotation schema with 13 label types for ear anatomy and Frank Sign
