"""Tests for clinical data loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.clinical_loader import (
    parse_turkish_decimal,
    parse_binary,
    parse_gender,
    parse_ef,
    parse_age,
    normalize_patient_id,
    extract_patient_id_from_image,
    ClinicalDataLoader,
    PatientRecord,
)


# ============================================================
# PARSING FUNCTION TESTS
# ============================================================

class TestTurkishDecimalParsing:
    """Test Turkish decimal format parsing."""
    
    def test_comma_decimal(self):
        """Parse comma as decimal separator."""
        assert parse_turkish_decimal("1,26") == pytest.approx(1.26)
        assert parse_turkish_decimal("0,93") == pytest.approx(0.93)
    
    def test_integer_values(self):
        """Parse integer values."""
        assert parse_turkish_decimal("95") == pytest.approx(95.0)
        assert parse_turkish_decimal(100) == pytest.approx(100.0)
    
    def test_missing_values(self):
        """Handle missing value indicators."""
        assert parse_turkish_decimal("-") is None
        assert parse_turkish_decimal("") is None
        assert parse_turkish_decimal(" ") is None
        assert parse_turkish_decimal(None) is None
    
    def test_special_text_values(self):
        """Handle special text values like 'H' for high."""
        assert parse_turkish_decimal("H") is None
        assert parse_turkish_decimal("L") is None
    
    def test_range_indicators(self):
        """Handle values with < or > prefix."""
        assert parse_turkish_decimal("<1,6") == pytest.approx(1.6)
        assert parse_turkish_decimal(">50000") == pytest.approx(50000.0)


class TestBinaryParsing:
    """Test binary column parsing (0/1)."""
    
    def test_valid_binary(self):
        """Parse valid binary values."""
        assert parse_binary(0) == 0
        assert parse_binary(1) == 1
        assert parse_binary("0") == 0
        assert parse_binary("1") == 1
    
    def test_missing_binary(self):
        """Handle missing binary values."""
        assert parse_binary("") is None
        assert parse_binary("-") is None
        assert parse_binary(None) is None
    
    def test_float_binary(self):
        """Parse float representation of binary."""
        assert parse_binary(1.0) == 1
        assert parse_binary(0.0) == 0


class TestGenderParsing:
    """Test gender column parsing."""
    
    def test_male(self):
        """Parse male gender."""
        assert parse_gender("E") == "M"
        assert parse_gender("e") == "M"
    
    def test_female(self):
        """Parse female gender."""
        assert parse_gender("K") == "F"
        assert parse_gender("k") == "F"
    
    def test_missing(self):
        """Handle missing gender."""
        assert parse_gender(None) is None
        assert parse_gender("") is None


class TestEFParsing:
    """Test ejection fraction parsing."""
    
    def test_percentage_format(self):
        """Parse percentage format strings."""
        assert parse_ef("55%") == pytest.approx(55.0)
        assert parse_ef("60%") == pytest.approx(60.0)
        assert parse_ef("35%") == pytest.approx(35.0)
    
    def test_turkish_decimal_ef(self):
        """Parse EF with Turkish decimal."""
        # Note: EF values are typically integers, but handle edge cases
        assert parse_ef("47%") == pytest.approx(47.0)
    
    def test_invalid_ef(self):
        """Handle invalid EF values."""
        assert parse_ef("200%") is None  # Data entry error
        assert parse_ef("-%") is None
        assert parse_ef("-") is None
        assert parse_ef(None) is None


class TestAgeParsing:
    """Test age parsing."""
    
    def test_valid_age(self):
        """Parse valid ages."""
        assert parse_age(59) == 59
        assert parse_age("72") == 72
        assert parse_age(36.0) == 36
    
    def test_invalid_age(self):
        """Handle invalid ages."""
        assert parse_age(None) is None
        assert parse_age(-5) is None
        assert parse_age(150) is None


class TestPatientIdNormalization:
    """Test patient ID normalization."""
    
    def test_simple_id(self):
        """Parse simple numeric ID."""
        assert normalize_patient_id("1763794") == "1763794"
    
    def test_id_with_suffix(self):
        """Remove protocol suffixes."""
        assert normalize_patient_id("1763794_1") == "1763794"
        assert normalize_patient_id("1692185_2") == "1692185"
    
    def test_missing_id(self):
        """Handle missing ID."""
        assert normalize_patient_id(None) is None
        assert normalize_patient_id("") is None


class TestImagePatientIdExtraction:
    """Test extraction of patient ID from image filename."""
    
    def test_standard_format(self):
        """Extract from 'ID - Name.jpeg' format."""
        assert extract_patient_id_from_image("1763794 - Ahmet Yılmaz.jpeg") == "1763794"
    
    def test_no_extension(self):
        """Handle filenames without extension."""
        assert extract_patient_id_from_image("1763794 - Test Name") == "1763794"
    
    def test_complex_name(self):
        """Handle names with special characters."""
        assert extract_patient_id_from_image("911218 - Ali Kemal Karataş.jpeg") == "911218"
    
    def test_invalid_format(self):
        """Handle unrecognized format."""
        assert extract_patient_id_from_image("random_image.jpeg") is None


# ============================================================
# PATIENT RECORD TESTS
# ============================================================

class TestPatientRecord:
    """Test PatientRecord dataclass."""
    
    def test_has_frank_sign_any(self):
        """Test any Frank Sign detection."""
        # Right only
        p1 = PatientRecord(patient_id="1", fs_right=1, fs_left=0)
        assert p1.has_frank_sign_any is True
        
        # Left only
        p2 = PatientRecord(patient_id="2", fs_right=0, fs_left=1)
        assert p2.has_frank_sign_any is True
        
        # Neither
        p3 = PatientRecord(patient_id="3", fs_right=0, fs_left=0)
        assert p3.has_frank_sign_any is False
    
    def test_has_frank_sign_bilateral(self):
        """Test bilateral Frank Sign detection."""
        # Bilateral
        p1 = PatientRecord(patient_id="1", fs_right=1, fs_left=1)
        assert p1.has_frank_sign_bilateral is True
        
        # Unilateral
        p2 = PatientRecord(patient_id="2", fs_right=1, fs_left=0)
        assert p2.has_frank_sign_bilateral is False
    
    def test_cv_risk_factor_count(self):
        """Test risk factor counting."""
        p = PatientRecord(
            patient_id="1",
            hypertension=1,
            diabetes=1,
            smoking=0,
            family_history=1
        )
        assert p.cv_risk_factor_count == 3
    
    def test_age_group(self):
        """Test age group categorization."""
        young = PatientRecord(patient_id="1", age=45)
        assert young.age_group == "young"
        
        middle = PatientRecord(patient_id="2", age=58)
        assert middle.age_group == "middle"
        
        elderly = PatientRecord(patient_id="3", age=72)
        assert elderly.age_group == "elderly"
    
    def test_ef_category(self):
        """Test EF category classification."""
        preserved = PatientRecord(patient_id="1", ef=55)
        assert preserved.ef_category == "preserved"
        
        mid_range = PatientRecord(patient_id="2", ef=45)
        assert mid_range.ef_category == "mid_range"
        
        reduced = PatientRecord(patient_id="3", ef=30)
        assert reduced.ef_category == "reduced"


# ============================================================
# CLINICAL DATA LOADER TESTS (require actual CSV)
# ============================================================

class TestClinicalDataLoader:
    """Tests for ClinicalDataLoader class."""
    
    @pytest.fixture
    def csv_path(self):
        """Get path to clinical data CSV."""
        return Path(__file__).parent.parent / "FS - AI - Sayfa1.csv"
    
    def test_loader_loads_file(self, csv_path):
        """Loader should load CSV without errors."""
        if not csv_path.exists():
            pytest.skip(f"Clinical data not found: {csv_path}")
        
        loader = ClinicalDataLoader(csv_path)
        df = loader.load()
        
        assert df is not None
        assert len(df) > 0
    
    def test_loader_renames_columns(self, csv_path):
        """Loader should rename Turkish columns to English."""
        if not csv_path.exists():
            pytest.skip(f"Clinical data not found: {csv_path}")
        
        loader = ClinicalDataLoader(csv_path)
        df = loader.load()
        
        assert 'patient_id' in df.columns
        assert 'fs_left' in df.columns
        assert 'gender' in df.columns
    
    def test_loader_adds_derived_columns(self, csv_path):
        """Loader should add computed columns."""
        if not csv_path.exists():
            pytest.skip(f"Clinical data not found: {csv_path}")
        
        loader = ClinicalDataLoader(csv_path)
        df = loader.load()
        
        assert 'has_fs_any' in df.columns
        assert 'cv_risk_count' in df.columns
    
    def test_summary_stats(self, csv_path):
        """Loader should generate summary statistics."""
        if not csv_path.exists():
            pytest.skip(f"Clinical data not found: {csv_path}")
        
        loader = ClinicalDataLoader(csv_path)
        stats = loader.get_summary_stats()
        
        assert 'total_patients' in stats
        assert stats['total_patients'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
