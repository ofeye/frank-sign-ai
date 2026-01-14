"""Clinical data loader and processor for Frank Sign project.

This module handles:
- Loading clinical CSV data
- Turkish decimal/format conversion
- Data cleaning and validation
- Linking clinical records to images
- Feature engineering

Example:
    >>> loader = ClinicalDataLoader("FS - AI - Sayfa1.csv")
    >>> df = loader.load()
    >>> print(f"Loaded {len(df)} patients")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

import pandas as pd
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

# Column name mappings (Turkish -> English)
COLUMN_MAPPING = {
    'HASTA ADI': 'patient_name',
    'DOSYA NUMARASI': 'patient_id',
    'PROTOKOL': 'protocol',
    'FS-SAĞ': 'fs_right',
    'FS - SOL': 'fs_left',
    'CİNSİYET': 'gender',
    'YAŞ': 'age',
    'HT': 'hypertension',
    'SİSTOLİK KB': 'systolic_bp',
    'DM': 'diabetes',
    'HDL': 'hdl',
    'NonHDL': 'non_hdl',
    'LDL': 'ldl',
    'TOTAL KOLESTEROL': 'total_cholesterol',
    'TRİGLİSERİT': 'triglycerides',
    'KREATİNİN': 'creatinine',
    'GFR': 'gfr',
    'TROPONIN': 'troponin',
    'SİGARA': 'smoking',
    'AÖ': 'family_history',
    'BMI': 'bmi',
    'EF': 'ef',
    'AORT KAPAK VEL': 'aortic_valve_velocity',
    'SYNTAX SCORE': 'syntax_score',
    'ASVCD': 'ascvd',
    'FRAMINGHAM': 'framingham',
    'BASKIN KORONER': 'dominant_coronary',
    'CVAT': 'cvat_status',
}

# Binary columns
BINARY_COLUMNS = ['hypertension', 'diabetes', 'smoking', 'family_history', 'fs_right', 'fs_left']

# Numeric columns with Turkish decimal format
TURKISH_DECIMAL_COLUMNS = [
    'hdl', 'non_hdl', 'ldl', 'total_cholesterol', 'triglycerides',
    'creatinine', 'aortic_valve_velocity'
]


# ============================================================
# PARSING FUNCTIONS
# ============================================================

def parse_turkish_decimal(value: Any) -> Optional[float]:
    """Convert Turkish decimal format to float.
    
    Args:
        value: String or numeric value, may use comma as decimal separator.
        
    Returns:
        Float value or None if parsing fails.
        
    Examples:
        >>> parse_turkish_decimal("1,26")
        1.26
        >>> parse_turkish_decimal("-")
        None
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip()
    
    # Handle special missing values
    if value_str in ['-', '', ' ', 'H', 'L']:
        return None
    
    # Handle range indicators
    if value_str.startswith('<') or value_str.startswith('>'):
        # Extract numeric part
        numeric_part = re.sub(r'[<>]', '', value_str)
        value_str = numeric_part
    
    try:
        # Replace Turkish comma with period
        value_str = value_str.replace(',', '.')
        return float(value_str)
    except (ValueError, AttributeError):
        return None


def parse_binary(value: Any) -> Optional[int]:
    """Parse binary column (0/1/empty).
    
    Returns:
        0, 1, or None if missing.
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip()
    if value_str in ['', ' ', '-']:
        return None
    
    try:
        return int(float(value_str))
    except (ValueError, TypeError):
        return None


def parse_gender(value: Any) -> Optional[str]:
    """Parse gender column.
    
    Args:
        value: 'E' for male, 'K' for female
        
    Returns:
        'M' for male, 'F' for female, or None
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip().upper()
    if value_str == 'E':
        return 'M'
    elif value_str == 'K':
        return 'F'
    return None


def parse_ef(value: Any) -> Optional[float]:
    """Parse ejection fraction percentage.
    
    Args:
        value: String like "55%", "60%", etc.
        
    Returns:
        Float percentage (0-100) or None if invalid.
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip()
    
    # Handle missing/invalid values
    if value_str in ['-', '-%', '', ' ']:
        return None
    
    # Handle obvious data errors (EF > 100%)
    if '200%' in value_str:
        return None
    
    # Remove % and parse
    try:
        numeric_str = value_str.replace('%', '').replace(',', '.').strip()
        ef = float(numeric_str)
        
        # Validate range
        if 0 <= ef <= 100:
            return ef
        return None
    except (ValueError, AttributeError):
        return None


def parse_age(value: Any) -> Optional[int]:
    """Parse age column."""
    if pd.isna(value):
        return None
    
    try:
        age = int(float(value))
        if 0 < age < 120:  # Reasonable age range
            return age
        return None
    except (ValueError, TypeError):
        return None


def normalize_patient_id(value: Any) -> Optional[str]:
    """Normalize patient ID (file number).
    
    Removes protocol suffixes like _1, _2.
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip()
    if not value_str:
        return None
    
    # Remove potential suffixes
    base_id = value_str.split('_')[0]
    
    # Remove any non-numeric characters except for specific patterns
    return base_id


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PatientRecord:
    """Single patient clinical record."""
    patient_id: str
    patient_name: Optional[str] = None
    
    # Frank Sign
    fs_right: Optional[int] = None
    fs_left: Optional[int] = None
    
    # Demographics
    gender: Optional[str] = None
    age: Optional[int] = None
    bmi: Optional[float] = None
    
    # Risk factors
    hypertension: Optional[int] = None
    diabetes: Optional[int] = None
    smoking: Optional[int] = None
    family_history: Optional[int] = None
    systolic_bp: Optional[int] = None
    
    # Lipid panel
    hdl: Optional[float] = None
    ldl: Optional[float] = None
    total_cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None
    
    # Cardiac
    ef: Optional[float] = None
    troponin: Optional[str] = None
    creatinine: Optional[float] = None
    gfr: Optional[float] = None
    
    # Scores
    syntax_score: Optional[float] = None
    framingham: Optional[float] = None
    
    @property
    def has_frank_sign_any(self) -> bool:
        """True if Frank Sign present on either ear."""
        return self.fs_right == 1 or self.fs_left == 1
    
    @property
    def has_frank_sign_bilateral(self) -> bool:
        """True if Frank Sign present on both ears."""
        return self.fs_right == 1 and self.fs_left == 1
    
    @property
    def cv_risk_factor_count(self) -> int:
        """Count of present cardiovascular risk factors."""
        factors = [self.hypertension, self.diabetes, self.smoking, self.family_history]
        return sum(1 for f in factors if f == 1)
    
    @property
    def age_group(self) -> Optional[str]:
        """Age category."""
        if self.age is None:
            return None
        if self.age < 50:
            return 'young'
        elif self.age < 65:
            return 'middle'
        else:
            return 'elderly'
    
    @property
    def ef_category(self) -> Optional[str]:
        """EF category (preserved/mid-range/reduced)."""
        if self.ef is None:
            return None
        if self.ef >= 50:
            return 'preserved'
        elif self.ef >= 40:
            return 'mid_range'
        else:
            return 'reduced'


# ============================================================
# DATA LOADER CLASS
# ============================================================

class ClinicalDataLoader:
    """Load and process clinical data from CSV.
    
    Attributes:
        csv_path: Path to the clinical data CSV file.
        
    Example:
        >>> loader = ClinicalDataLoader("data/clinical/FS - AI - Sayfa1.csv")
        >>> df = loader.load()
        >>> patients = loader.to_patient_records(df)
    """
    
    def __init__(self, csv_path: str | Path):
        """Initialize loader with CSV path.
        
        Args:
            csv_path: Path to clinical CSV file.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Clinical data file not found: {self.csv_path}")
    
    def load(self, rename_columns: bool = True) -> pd.DataFrame:
        """Load and clean clinical data.
        
        Args:
            rename_columns: If True, rename Turkish columns to English.
            
        Returns:
            Cleaned pandas DataFrame.
        """
        # Load CSV
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        # Rename columns
        if rename_columns:
            df = df.rename(columns=COLUMN_MAPPING)
        
        # Clean data
        df = self._clean_data(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all data cleaning transformations."""
        df = df.copy()
        
        # Parse Turkish decimals
        for col in TURKISH_DECIMAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(parse_turkish_decimal)
        
        # Parse binary columns
        for col in BINARY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(parse_binary)
        
        # Parse specific columns
        if 'gender' in df.columns:
            df['gender'] = df['gender'].apply(parse_gender)
        
        if 'age' in df.columns:
            df['age'] = df['age'].apply(parse_age)
        
        if 'ef' in df.columns:
            df['ef'] = df['ef'].apply(parse_ef)
        
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].apply(normalize_patient_id)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/computed columns."""
        # Frank Sign any
        if 'fs_right' in df.columns and 'fs_left' in df.columns:
            df['has_fs_any'] = ((df['fs_right'] == 1) | (df['fs_left'] == 1)).astype(int)
            df['has_fs_bilateral'] = ((df['fs_right'] == 1) & (df['fs_left'] == 1)).astype(int)
        
        # CV risk factor count
        risk_cols = ['hypertension', 'diabetes', 'smoking', 'family_history']
        existing_risk_cols = [c for c in risk_cols if c in df.columns]
        if existing_risk_cols:
            df['cv_risk_count'] = df[existing_risk_cols].sum(axis=1, skipna=True)
        
        # Lipid ratio
        if 'total_cholesterol' in df.columns and 'hdl' in df.columns:
            df['lipid_ratio'] = df['total_cholesterol'] / df['hdl'].replace(0, np.nan)
        
        # Age group
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 50, 65, 120],
                labels=['young', 'middle', 'elderly']
            )
        
        # EF category
        if 'ef' in df.columns:
            df['ef_category'] = pd.cut(
                df['ef'],
                bins=[0, 40, 50, 100],
                labels=['reduced', 'mid_range', 'preserved']
            )
        
        return df
    
    def to_patient_records(self, df: Optional[pd.DataFrame] = None) -> List[PatientRecord]:
        """Convert DataFrame to list of PatientRecord objects.
        
        Args:
            df: DataFrame to convert. If None, loads from CSV.
            
        Returns:
            List of PatientRecord objects.
        """
        if df is None:
            df = self.load()
        
        records = []
        for _, row in df.iterrows():
            if pd.isna(row.get('patient_id')):
                continue
            
            record = PatientRecord(
                patient_id=str(row['patient_id']),
                patient_name=row.get('patient_name'),
                fs_right=row.get('fs_right'),
                fs_left=row.get('fs_left'),
                gender=row.get('gender'),
                age=row.get('age'),
                bmi=row.get('bmi'),
                hypertension=row.get('hypertension'),
                diabetes=row.get('diabetes'),
                smoking=row.get('smoking'),
                family_history=row.get('family_history'),
                systolic_bp=row.get('systolic_bp'),
                hdl=row.get('hdl'),
                ldl=row.get('ldl'),
                total_cholesterol=row.get('total_cholesterol'),
                triglycerides=row.get('triglycerides'),
                ef=row.get('ef'),
                troponin=str(row.get('troponin')) if pd.notna(row.get('troponin')) else None,
                creatinine=row.get('creatinine'),
                gfr=row.get('gfr'),
                syntax_score=row.get('syntax_score'),
                framingham=row.get('framingham'),
            )
            records.append(record)
        
        return records
    
    def get_summary_stats(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate summary statistics for the dataset.
        
        Returns:
            Dictionary with key statistics.
        """
        if df is None:
            df = self.load()
        
        stats = {
            'total_patients': len(df),
            'with_fs_right': df['fs_right'].sum() if 'fs_right' in df.columns else 0,
            'with_fs_left': df['fs_left'].sum() if 'fs_left' in df.columns else 0,
            'with_fs_any': df['has_fs_any'].sum() if 'has_fs_any' in df.columns else 0,
            'male_count': (df['gender'] == 'M').sum() if 'gender' in df.columns else 0,
            'female_count': (df['gender'] == 'F').sum() if 'gender' in df.columns else 0,
            'age_mean': df['age'].mean() if 'age' in df.columns else None,
            'age_std': df['age'].std() if 'age' in df.columns else None,
            'missing_fs_right': df['fs_right'].isna().sum() if 'fs_right' in df.columns else 0,
            'missing_fs_left': df['fs_left'].isna().sum() if 'fs_left' in df.columns else 0,
        }
        
        return stats


# ============================================================
# IMAGE-CLINICAL LINKER
# ============================================================

def extract_patient_id_from_image(image_name: str) -> Optional[str]:
    """Extract patient ID from image filename.
    
    Expected format: "{patient_id} - {patient_name}.jpeg"
    or: "{patient_name}-{patient_id}.jpeg"
    
    Args:
        image_name: Image filename.
        
    Returns:
        Extracted patient ID or None.
    """
    if not image_name:
        return None
    
    # Remove extension
    name = Path(image_name).stem
    
    # Try format: "1763794 - Name"
    if ' - ' in name:
        parts = name.split(' - ')
        # Check if first part is numeric (patient ID)
        if parts[0].strip().replace('_', '').isdigit():
            return parts[0].strip().split('_')[0]
        # Check if last part is numeric
        if parts[-1].strip().replace('_', '').isdigit():
            return parts[-1].strip().split('_')[0]
    
    # Try format: "Name, ID"
    if ', ' in name:
        parts = name.split(', ')
        for part in parts:
            if part.strip().isdigit():
                return part.strip()
    
    # Try format with hyphen
    if '-' in name:
        parts = name.split('-')
        for part in parts:
            if part.strip().isdigit():
                return part.strip()
    
    return None


def link_clinical_to_images(
    clinical_df: pd.DataFrame,
    image_names: List[str]
) -> Dict[str, Optional[pd.Series]]:
    """Link images to clinical records.
    
    Args:
        clinical_df: Clinical data DataFrame.
        image_names: List of image filenames.
        
    Returns:
        Dict mapping image name to clinical record (or None if not found).
    """
    results = {}
    
    for image_name in image_names:
        patient_id = extract_patient_id_from_image(image_name)
        
        if patient_id is None:
            results[image_name] = None
            continue
        
        # Find matching record
        mask = clinical_df['patient_id'].astype(str) == str(patient_id)
        matches = clinical_df[mask]
        
        if len(matches) > 0:
            results[image_name] = matches.iloc[0]
        else:
            results[image_name] = None
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "FS - AI - Sayfa1.csv"
    
    try:
        loader = ClinicalDataLoader(csv_path)
        df = loader.load()
        stats = loader.get_summary_stats(df)
        
        print(f"✅ Loaded {stats['total_patients']} patients")
        print(f"👤 Male: {stats['male_count']}, Female: {stats['female_count']}")
        print(f"📊 Age: {stats['age_mean']:.1f} ± {stats['age_std']:.1f}")
        print(f"🔍 Frank Sign (right): {stats['with_fs_right']}")
        print(f"🔍 Frank Sign (left): {stats['with_fs_left']}")
        print(f"🔍 Frank Sign (any): {stats['with_fs_any']}")
        print(f"❓ Missing FS-right: {stats['missing_fs_right']}")
        print(f"❓ Missing FS-left: {stats['missing_fs_left']}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
