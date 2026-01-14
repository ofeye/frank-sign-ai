# Data Management Guide

> Documentation for managing sample data, production data transitions, and data versioning.

## Current Data Status

| Data Source | Type | Status | Records | Notes |
|-------------|------|--------|---------|-------|
| `FS - AI - Sayfa1.csv` | Clinical | 📌 **SAMPLE** | 204 | Test data for development |
| `annotations.xml` | CVAT | 📌 **PILOT** | 121 images | Pilot annotation batch |
| `data/raw/` | Images | ⏳ Pending | TBD | Full image dataset |

> [!WARNING]
> **Sample Data Notice**: `FS - AI - Sayfa1.csv` is a development sample. Production data will be provided separately with more complete records.

---

## Data Transition Plan

### Phase 1: Development (Current)
- Using sample CSV (204 patients)
- Using pilot CVAT annotations (121 images)
- Code designed to be data-agnostic

### Phase 2: Expanded Data
When production data arrives:

1. **Clinical CSV Update**
   ```bash
   # Place new CSV in data/clinical/
   mv new_production_data.csv data/clinical/production_data.csv
   
   # Update config
   # configs/default.yaml → data.clinical_data_path
   ```

2. **Verify Column Compatibility**
   ```python
   from franksign.data import ClinicalDataLoader
   loader = ClinicalDataLoader("data/clinical/production_data.csv")
   df = loader.load()
   print(df.columns)  # Verify all expected columns exist
   ```

3. **Run Data Validation**
   ```bash
   python scripts/validate_data.py --clinical data/clinical/production_data.csv
   ```

### Phase 3: Full Production
- Complete patient cohort (1000+ expected)
- All images annotated
- Clinical outcomes linked

---

## Configuration for Different Data Sources

### Sample/Development (default.yaml)
```yaml
data:
  clinical_data_path: "FS - AI - Sayfa1.csv"  # Sample
  annotations_path: "data/annotations/annotations.xml"
```

### Production (production.yaml)
```yaml
data:
  clinical_data_path: "data/clinical/production_data.csv"
  annotations_path: "data/annotations/production_annotations.xml"
```

### Usage
```bash
# Development
python scripts/train.py --config configs/default.yaml

# Production  
python scripts/train.py --config configs/production.yaml
```

---

## Expected Data Changes

### Clinical CSV Evolution

| Current Column | Status | Potential Changes |
|----------------|--------|-------------------|
| All 27 columns | ✅ Defined | May add more lab values |
| SYNTAX SCORE | Sparse | Will be filled for more patients |
| FRAMINGHAM | Sparse | Will be calculated |
| CVAT | Incomplete | Will link to image annotations |

### New Columns (Anticipated)

| Column | Purpose | Phase |
|--------|---------|-------|
| `SYNTAX_CATEGORY` | Low/Mid/High severity | Phase 2 |
| `CAD_SEVERITY` | Composite risk score | Phase 2 |
| `FOLLOW_UP_EVENT` | Outcome tracking | Phase 5 |
| `IMAGE_LEFT_PATH` | Direct image link | Phase 2 |
| `IMAGE_RIGHT_PATH` | Direct image link | Phase 2 |

---

## Data Versioning with DVC

When data grows large:

```bash
# Initialize DVC (one-time)
dvc init
dvc remote add -d storage /path/to/storage

# Track data files
dvc add data/clinical/production_data.csv
dvc add data/raw/

# Push to remote
dvc push
```

---

## Linking Images to Clinical Records

### Current Convention
Image filename: `{DOSYA_NUMARASI} - {HASTA_ADI}.jpeg`

### Linking Code
```python
from franksign.data.clinical_loader import extract_patient_id_from_image

# From image to clinical record
patient_id = extract_patient_id_from_image("1763794 - Ahmet Yılmaz.jpeg")
clinical_record = clinical_df[clinical_df['patient_id'] == patient_id]
```

### Future Enhancement
- Add `IMAGE_PATH` column to clinical CSV
- Or store mapping in separate JSON file

---

## Data Quality Checks

Before using any new dataset:

```python
def validate_clinical_data(df):
    """Validate clinical dataset."""
    checks = []
    
    # Required columns
    required = ['patient_id', 'fs_left', 'gender', 'age']
    for col in required:
        checks.append((col in df.columns, f"Missing column: {col}"))
    
    # Value ranges
    if 'age' in df.columns:
        checks.append((df['age'].min() > 0, "Age has invalid values"))
        checks.append((df['age'].max() < 120, "Age has outliers"))
    
    # Report
    for passed, msg in checks:
        print(f"{'✅' if passed else '❌'} {msg}")
```
