# Clinical Data Schema

> Complete documentation of clinical dataset variables for Frank Sign AI Project

## Dataset Overview

- **Source**: Hospital cardiology clinic database
- **Patients**: 204 total (as of latest export)
- **File**: `FS - AI - Sayfa1.csv`
- **Encoding**: UTF-8 with Turkish characters

---

## Patient Identification

| Column | Turkish Name | Type | Description | Example |
|--------|--------------|------|-------------|---------|
| `HASTA ADI` | Hasta Adı | string | Patient full name (anonymize before processing) | "Ahmet Yılmaz" |
| `DOSYA NUMARASI` | Dosya Numarası | string | Hospital file number (primary ID) | "1763794" |
| `PROTOKOL` | Protokol | string | Visit protocol number (may include suffix) | "1763794_1" |

> [!WARNING]
> Patient names must be anonymized before any processing. Use only `DOSYA NUMARASI` as identifier in code.

---

## Frank Sign Assessment (Target Variables)

| Column | Name | Type | Values | Description |
|--------|------|------|--------|-------------|
| `FS-SAĞ` | Frank Sign Right | binary/empty | 0, 1, empty | Frank Sign presence on right ear |
| `FS - SOL` | Frank Sign Left | binary/empty | 0, 1, empty | Frank Sign presence on left ear |
| `CVAT` | CVAT Status | string/empty | TBD | Whether CVAT annotation exists |

### Interpretation
- `0` = Frank Sign absent
- `1` = Frank Sign present
- `empty` = Not assessed or missing data

### Derived Variables
- `has_frank_sign_any`: True if FS-SAĞ=1 OR FS-SOL=1
- `has_frank_sign_bilateral`: True if FS-SAĞ=1 AND FS-SOL=1

---

## Demographics

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `CİNSİYET` | categorical | E (Erkek), K (Kadın) | Gender: Male (E) / Female (K) |
| `YAŞ` | integer | 36-82 | Age in years |

---

## Cardiovascular Risk Factors

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `HT` | binary | 0, 1, empty | Hypertension |
| `SİSTOLİK KB` | integer | 105-145 | Systolic blood pressure (mmHg) |
| `DM` | binary | 0, 1, empty | Diabetes Mellitus |
| `SİGARA` | binary | 0, 1, empty | Smoking status (current/former) |
| `AÖ` | binary | 0, 1, empty | Family history (Aile Öyküsü) |
| `BMI` | integer | 23-38 | Body Mass Index |

---

## Lipid Panel

| Column | Type | Unit | Normal Range | Description |
|--------|------|------|--------------|-------------|
| `HDL` | float | mg/dL | >40 (M), >50 (F) | HDL Cholesterol |
| `NonHDL` | float | mg/dL | <130 | Non-HDL Cholesterol |
| `LDL` | float | mg/dL | <100 | LDL Cholesterol |
| `TOTAL KOLESTEROL` | float | mg/dL | <200 | Total Cholesterol |
| `TRİGLİSERİT` | float | mg/dL | <150 | Triglycerides |

### Data Quality Notes
- Some values are marked as "-" (missing)
- Decimal separator is Turkish format: "," instead of "."
- Must convert "0,93" → 0.93 during parsing

---

## Kidney Function

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `KREATİNİN` | float | mg/dL | Serum creatinine |
| `GFR` | integer | mL/min/1.73m² | Glomerular Filtration Rate |

---

## Cardiac Markers

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `TROPONIN` | string | ng/L | Cardiac troponin (may contain text like "H", ">50000") |
| `EF` | string | % | Ejection Fraction (e.g., "55%", "200%", "-%") |
| `AORT KAPAK VEL` | float | m/s | Aortic valve velocity |

### Troponin Value Handling
- Numeric values as strings with Turkish decimal: "1,6"
- Special values: "H" (high), "<1,6" (below threshold), ">50000" (above range)

### EF Value Handling
- Format: "55%" or percentage string
- Invalid values: "200%" (data entry error), "-%" (missing)
- Parse as integer, handle edge cases

---

## Coronary Artery Disease Scoring

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `SYNTAX SCORE` | float | 0-100+ | Coronary lesion complexity score |
| `ASVCD` | string | TBD | ASCVD risk score |
| `FRAMINGHAM` | float | TBD | Framingham Risk Score |
| `BASKIN KORONER` | string | TBD | Dominant coronary artery |

---

## Data Cleaning Requirements

### 1. Turkish Decimal Conversion
```python
def parse_turkish_decimal(value: str) -> float:
    """Convert Turkish decimal format to float."""
    if pd.isna(value) or value in ['-', '', ' ']:
        return np.nan
    return float(str(value).replace(',', '.'))
```

### 2. Binary Column Parsing
```python
def parse_binary(value) -> Optional[int]:
    """Parse binary columns (0/1/empty)."""
    if pd.isna(value) or value == '':
        return None
    return int(value)
```

### 3. EF Parsing
```python
def parse_ef(value: str) -> Optional[float]:
    """Parse ejection fraction percentage."""
    if pd.isna(value) or value in ['-', '-%', '200%']:
        return None
    # Remove % and convert
    return float(str(value).replace('%', '').replace(',', '.'))
```

### 4. Patient ID Normalization
```python
def normalize_patient_id(dosya_no: str) -> str:
    """Normalize patient file number."""
    # Remove any _1, _2 suffixes from protocol numbers
    return str(dosya_no).split('_')[0] if dosya_no else ''
```

---

## Missing Data Statistics (Estimated)

| Column | Missing % | Notes |
|--------|-----------|-------|
| FS-SAĞ | ~60% | Many not assessed |
| FS-SOL | ~40% | Primary assessment column |
| YAŞ | ~5% | Some rows incomplete |
| Lipid panel | ~10% | Lab results pending |
| EF | ~15% | Echo not performed |
| SYNTAX | ~80% | Only for CAD patients |

---

## Data Merging Strategy

### Linking Clinical Data to Images

1. **Primary Key**: `DOSYA NUMARASI` in clinical CSV
2. **Image Naming Convention**: `{DOSYA_NUMARASI} - {HASTA_ADI}.jpeg`
3. **CVAT patient_metadata**: Contains `patient_id` attribute

```python
def link_clinical_to_image(clinical_df, image_name):
    """Link image to clinical record by patient ID."""
    # Extract patient ID from image name
    # Format: "1763794 - Halil İbrahim Kaymak.jpeg"
    patient_id = image_name.split(' - ')[0].strip()
    
    # Find matching clinical record
    match = clinical_df[clinical_df['DOSYA NUMARASI'] == patient_id]
    return match.iloc[0] if len(match) > 0 else None
```

---

## Feature Engineering for ML

### Derived Features

| Feature | Formula | Type |
|---------|---------|------|
| `has_fs_any` | FS-SAĞ=1 OR FS-SOL=1 | binary |
| `has_fs_bilateral` | FS-SAĞ=1 AND FS-SOL=1 | binary |
| `age_group` | <50: young, 50-65: middle, >65: elderly | categorical |
| `lipid_ratio` | TOTAL KOLESTEROL / HDL | float |
| `cv_risk_count` | sum(HT, DM, SİGARA, AÖ) | 0-4 |
| `gfr_category` | GFR stages: ≥90, 60-89, 30-59, <30 | categorical |
| `ef_category` | preserved (≥50), mid-range (40-49), reduced (<40) | categorical |

### Target Variable Options

1. **Binary Classification**: `has_frank_sign_any` → CAD presence (via SYNTAX>0)
2. **Regression**: Frank Sign features → SYNTAX Score
3. **Multi-label**: Predict multiple CVD risk factors

---

## Data Version Control

Track changes to the clinical dataset:

| Version | Date | Records | Notes |
|---------|------|---------|-------|
| v1.0 | 2026-01-13 | 204 | Initial export from hospital |
| v1.1 | TBD | TBD | After data cleaning |
