"""Data loading and preprocessing modules."""

from franksign.data.cvat_parser import CVATParser
from franksign.data.geometric_features import GeometricFeatureExtractor
from franksign.data.clinical_loader import ClinicalDataLoader, PatientRecord
from franksign.data.preprocess import preprocess_images
from franksign.data.validation import ClinicalSchema, ValidationIssue, validate_cvat_project

try:
    from franksign.data.dataset import FrankSignDataset, Sample
    _HAS_TORCH = True
except ImportError:
    FrankSignDataset = None  # type: ignore
    Sample = None  # type: ignore
    _HAS_TORCH = False

__all__ = [
    "CVATParser",
    "GeometricFeatureExtractor",
    "ClinicalDataLoader",
    "PatientRecord",
    "preprocess_images",
    "ClinicalSchema",
    "ValidationIssue",
    "validate_cvat_project",
]

if _HAS_TORCH:
    __all__.extend(["FrankSignDataset", "Sample"])
