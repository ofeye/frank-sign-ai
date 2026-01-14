# AI Agent Guidelines for Frank Sign Project

> This document provides context and guidelines for AI coding assistants working on this project.

## 🎯 Project Context

**Domain**: Medical Image Analysis + Cardiovascular Risk Prediction

**Goal**: Segment Frank's Sign (diagonal earlobe crease) from ear images and extract geometric features to predict cardiovascular disease risk.

**Key Constraint**: This is a research project with evolving requirements. Code should be modular and easily modifiable.

---

## ⚠️ CRITICAL: Agent Responsibilities

> [!IMPORTANT]
> Every AI agent working on this project MUST follow these guidelines:

### 1. Continuous Documentation Monitoring
- **ALWAYS read** ROADMAP.md and task.md before starting work
- **CHECK** CHANGELOG.md to understand recent changes
- **UPDATE** documentation when making changes
- **PROPOSE** improvements if you spot better approaches

### 2. Proactive Improvement Suggestions
When you notice:
- Inefficient code patterns → Suggest refactoring
- Missing tests → Propose test cases
- Documentation gaps → Fill them
- Better libraries/approaches → Ask user before switching

### 3. Sample vs Production Data Awareness
> [!WARNING]
> Current data is SAMPLE DATA for development:
> - `FS - AI - Sayfa1.csv` → Sample (204 patients)
> - `annotations.xml` → Pilot (121 images)
> 
> See `docs/data_management.md` for production data transition plan.

---

## � MUST: Documentation Sync Protocol

> [!CAUTION]
> **After EVERY change, you MUST update the relevant documentation.**
> This is NON-NEGOTIABLE. Failure to do this causes tracking loss across agents.

### After Completing ANY Task:

```
✅ CHECKLIST (Copy this into your thinking):
□ 1. CHANGELOG.md → Add entry under [Unreleased]
□ 2. ROADMAP.md → Mark items [x] or update status
□ 3. docs/processes/*.md → Update if parallel process affected
□ 4. AGENTS.md → Update if new patterns/rules introduced
□ 5. docs/audit_report_draft.md → Mark completed action items
```

### When to Create NEW Documentation:

| Situation | Action |
|-----------|--------|
| New major feature or subsystem | Create `docs/[feature].md` |
| New parallel work stream | Create `docs/processes/[process].md` |
| Complex multi-day task | Create dedicated tracking doc |
| Breaking change or migration | Add to `docs/migrations/` |

### Documentation Update Examples:

**Small change (e.g., bug fix):**
```markdown
# CHANGELOG.md
### Fixed
- CVAT parser: handle semicolon-separated points
```

**New feature (e.g., logging system):**
```markdown
# CHANGELOG.md
### Added
- Package-wide logging configuration (`src/franksign/__init__.py`)
- Pre-commit hooks (Black, Ruff, isort)

# Ayrıca güncelle: ROADMAP.md, task.md, environment.md
```

### Verification Command:
Before ending your session, mentally run:
```
"Did I update CHANGELOG? ROADMAP? Any process docs?"
```

---

## �📋 Important Files to Reference

| File | Purpose | When to Read |
|------|---------|--------------|
| `Project_Main.md` | Full TÜBİTAK proposal (Turkish) | Understanding project goals |
| `ROADMAP.md` | Project phases and timeline | **Before starting any task** |
| `CHANGELOG.md` | Version history | **Before making changes** |
| `docs/data_management.md` | Sample vs production data | Working on data pipeline |
| `docs/environment.md` | Python environment setup | Environment issues |
| `docs/clinical_data_schema.md` | Clinical column definitions | Working with clinical data |
| `docs/data_schema.md` | CVAT annotation schema | Working with image annotations |
| `configs/default.yaml` | Default configuration | Training/evaluation |

---

## 🔄 Parallel Process Tracking

> [!TIP]
> When multiple work streams run in parallel, track each separately:

### Active Parallel Processes
Each major parallel workstream should have its own tracking document in `docs/processes/`:

| Process | Doc Location | Status |
|---------|--------------|--------|
| Data Pipeline | `docs/processes/data_pipeline.md` | Active |
| Model Development | `docs/processes/model_development.md` | Planned |
| Clinical Integration | `docs/processes/clinical_integration.md` | Planned |

### Creating a New Process Document
```markdown
# Process: [Name]

## Status: [Active/Paused/Complete]
## Last Updated: YYYY-MM-DD
## Owner: [Human/Agent]

## Current Objectives
- [ ] Objective 1
- [ ] Objective 2

## Progress Log
### YYYY-MM-DD
- What was done
- Blockers encountered
- Next steps

## Dependencies
- Depends on: [other processes]
- Blocking: [other processes]
```

---

## 🏗️ Code Style Preferences

### General
- **Language**: Python 3.11+ with type hints
- **Formatting**: Follow PEP 8, use `black` formatter
- **Docstrings**: Google style docstrings for all public functions
- **Testing**: pytest for unit tests

### Virtual Environment
```bash
# ALWAYS use project venv
source .venv/bin/activate

# Verify environment
python -c "import pandas; import numpy; print('✅ OK')"
```

See `docs/environment.md` for full environment documentation.

### Example Code Style
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np


@dataclass
class FrankSignFeatures:
    """Geometric features extracted from Frank Sign annotation.
    
    Attributes:
        length_mm: Arc length of the crease in millimeters.
        curvature_mean: Mean discrete curvature along the line.
        tortuosity: Ratio of arc length to euclidean distance (≥1.0).
    """
    length_mm: float
    curvature_mean: float
    tortuosity: float
```

---

## 🔧 Common Tasks

### 1. Adding New Annotation Label
When a new label is added to CVAT:
1. Update `src/franksign/data/cvat_parser.py` - Add parsing logic
2. Update `docs/data_schema.md` - Document the new label
3. Update `CHANGELOG.md` - Log the change

### 2. Adding New Clinical Column
When production data has new columns:
1. Update `src/franksign/data/clinical_loader.py` - Add parsing
2. Update `docs/clinical_data_schema.md` - Document
3. Update `CHANGELOG.md`

### 3. Running Experiments
```bash
# ALWAYS activate venv first
source .venv/bin/activate

# Use config files, don't hardcode hyperparameters
python scripts/train.py --config configs/experiments/exp001.yaml
```

---

## ⚠️ Decision Guidelines

### When Uncertain About Implementation:
1. **Check Project_Main.md** - Section 4 (YÖNTEM) has detailed methodology
2. **Prefer simplicity** - Start with basic implementation, refine later
3. **Keep flexibility** - Use config files, avoid hardcoding
4. **Document decisions** - Add comments explaining why, not just what
5. **ASK USER** if decision has significant impact

### When to Ask User:
- [ ] Adding new dependencies (especially large ones like PyTorch)
- [ ] Changing data schema or file structure
- [ ] Modifying core algorithms
- [ ] Any breaking changes
- [ ] Uncertainty about medical/clinical interpretation

### Sample Data Considerations:
- Current CSV is **sample data** - expect column additions/changes
- Parser should handle missing or unknown columns gracefully
- Always validate data before processing
- Log warnings for unexpected data patterns

---

## 📊 Annotation Labels Reference

Current labels in CVAT (from annotations.xml):

| Label | Type | Purpose |
|-------|------|---------|
| `ear_outer_contour` | polygon | Full ear boundary |
| `franks_sign_region` | polygon | Area containing Frank Sign |
| `franks_sign_line` | polyline | The crease itself |
| `ear_canal_center` | point | Reference point |
| `tragus_point` | point | Anatomical landmark |
| `antitragus_point` | point | Anatomical landmark |
| `earlobe_tip` | point | Lowest point of earlobe |
| `ear_top` | point | Highest point of ear |
| `intertragic_notch` | point | Notch between tragus/antitragus |
| `earlobe_attachment_point` | point | Where earlobe meets face |
| `image_quality_assessment` | point | Metadata carrier |
| `annotation_metadata` | point | Annotator info |
| `patient_metadata` | point | Patient ID, name |

---

## 🏥 Clinical Data Columns Reference

Key columns in `FS - AI - Sayfa1.csv` (SAMPLE DATA):

| Column | English Name | Type | Notes |
|--------|--------------|------|-------|
| `DOSYA NUMARASI` | patient_id | string | Primary key, links to images |
| `FS-SAĞ` | fs_right | binary | Frank Sign right ear (0/1) |
| `FS - SOL` | fs_left | binary | Frank Sign left ear (0/1) |
| `CİNSİYET` | gender | cat | E=Male, K=Female |
| `YAŞ` | age | int | Age in years |
| `HT` | hypertension | binary | Hypertension |
| `DM` | diabetes | binary | Diabetes Mellitus |
| `SİGARA` | smoking | binary | Smoking status |
| `EF` | ef | string | Ejection Fraction (e.g., "55%") |
| `SYNTAX SCORE` | syntax_score | float | Coronary lesion complexity |

> [!TIP]
> Use `ClinicalDataLoader` class to handle Turkish decimal format (comma → period) and missing values.

---

## 🚫 Things to Avoid

1. **Don't modify Project_Main.md** - It's the official TÜBİTAK document
2. **Don't hardcode paths** - Use configs or command-line arguments
3. **Don't skip type hints** - They're essential for code clarity
4. **Don't create notebooks in src/** - Notebooks go in `notebooks/` (if needed)
5. **Don't train without logging** - Always track experiments
6. **Don't assume production data** - Current data is sample only
7. **Don't modify data files directly** - Use scripts and version control

---

## ✅ Checklist Before Committing

- [ ] Virtual environment activated (`.venv`)
- [ ] Code passes `black` formatting
- [ ] Type hints on all function signatures
- [ ] Docstrings on all public functions
- [ ] Tests added for new functionality
- [ ] CHANGELOG.md updated
- [ ] No hardcoded paths or credentials
- [ ] Documentation updated if needed

---

## 🔄 Continuous Improvement Protocol

### After Each Major Task:
1. **Review** what was done
2. **Identify** potential improvements
3. **Document** learnings
4. **Propose** next steps to user

### Improvement Suggestion Format:
```markdown
## 💡 Improvement Suggestion

**Area**: [Code/Docs/Process/Architecture]
**Current State**: Brief description
**Proposed Change**: What to do
**Benefit**: Why it's better
**Effort**: Low/Medium/High
**Risk**: Low/Medium/High

Would you like me to implement this?
```
