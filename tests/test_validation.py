"""Tests for validation module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.cvat_parser import (
    Point,
    PolylineAnnotation,
    PolygonAnnotation,
    ImageAnnotations,
    CVATProject,
    LabelDefinition,
)
from franksign.data.validation import (
    ClinicalSchema,
    ValidationIssue,
    validate_cvat_project,
    _validate_image_annotations,
    _segments_intersect,
    _is_self_intersecting,
    EXPECTED_LABELS,
    REQUIRED_LABELS,
    MIN_POLYLINE_LENGTH_PX,
    MIN_POLYGON_AREA_PX2,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def valid_project():
    """CVAT project with valid labels and annotations."""
    labels = [LabelDefinition(name=label, color="#FF0000", type="polygon") for label in REQUIRED_LABELS]
    project = CVATProject(
        id=1, name="test", created="2026-01-01", updated="2026-01-01",
        labels=labels, images=[]
    )
    return project


@pytest.fixture
def valid_image():
    """Image with valid annotations."""
    img = ImageAnnotations(id=1, name="valid.jpg", width=200, height=200)
    
    # Valid ear contour polygon
    img.polygons.append(PolygonAnnotation(
        label="ear_outer_contour",
        points=[Point(10, 10), Point(100, 10), Point(100, 100), Point(10, 100)]
    ))
    
    # Valid Frank Sign line
    img.polylines.append(PolylineAnnotation(
        label="franks_sign_line",
        points=[Point(20, 50), Point(80, 50)],
        attributes={"presence": "present"}
    ))
    
    return img


# ============================================================
# SEGMENT INTERSECTION TESTS
# ============================================================

class TestSegmentsIntersect:
    """Tests for segment intersection detection."""
    
    def test_intersecting_segments(self):
        """Crossing segments should intersect."""
        # X pattern
        p1, p2 = (0, 0), (10, 10)
        p3, p4 = (0, 10), (10, 0)
        assert _segments_intersect(p1, p2, p3, p4) == True
    
    def test_non_intersecting_parallel(self):
        """Parallel segments don't intersect."""
        p1, p2 = (0, 0), (10, 0)
        p3, p4 = (0, 5), (10, 5)
        assert _segments_intersect(p1, p2, p3, p4) == False
    
    def test_non_intersecting_distant(self):
        """Distant segments don't intersect."""
        p1, p2 = (0, 0), (5, 0)
        p3, p4 = (10, 10), (15, 10)
        assert _segments_intersect(p1, p2, p3, p4) == False


class TestIsSelfIntersecting:
    """Tests for polygon self-intersection detection."""
    
    def test_simple_square_not_intersecting(self):
        """Simple square is not self-intersecting."""
        import numpy as np
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert _is_self_intersecting(square) == False
    
    def test_figure_eight_is_intersecting(self):
        """Figure-8 shape is self-intersecting."""
        import numpy as np
        # This forms a bowtie/figure-8
        bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]])
        assert _is_self_intersecting(bowtie) == True
    
    def test_triangle_not_intersecting(self):
        """Triangle is not self-intersecting."""
        import numpy as np
        triangle = np.array([[0, 0], [10, 0], [5, 10]])
        assert _is_self_intersecting(triangle) == False


# ============================================================
# CVAT PROJECT VALIDATION TESTS
# ============================================================

class TestValidateCVATProject:
    """Tests for validate_cvat_project function."""
    
    def test_valid_project_no_issues(self, valid_project, valid_image):
        """Valid project with good annotations has no errors."""
        valid_project.images.append(valid_image)
        issues = validate_cvat_project(valid_project)
        
        # Filter to errors only
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0
    
    def test_missing_required_labels(self):
        """Missing required labels triggers error."""
        # Project with no labels
        project = CVATProject(
            id=1, name="empty", created="2026-01-01", updated="2026-01-01",
            labels=[], images=[]
        )
        issues = validate_cvat_project(project)
        
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("Missing required labels" in i.message for i in errors)
    
    def test_unknown_labels_warning(self, valid_project):
        """Unknown labels trigger warning."""
        valid_project.labels.append(LabelDefinition(
            name="unknown_label", color="#00FF00", type="polygon"
        ))
        # labels already appended above
        issues = validate_cvat_project(valid_project)
        
        warnings = [i for i in issues if i.level == "warning"]
        assert any("Unknown labels" in i.message for i in warnings)


# ============================================================
# IMAGE ANNOTATION VALIDATION TESTS
# ============================================================

class TestValidateImageAnnotations:
    """Tests for _validate_image_annotations function."""
    
    def test_valid_image_no_issues(self, valid_image):
        """Valid image has no issues."""
        issues = list(_validate_image_annotations(valid_image))
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0
    
    def test_polyline_too_few_points(self):
        """Polyline with < 2 points triggers error."""
        img = ImageAnnotations(id=1, name="bad.jpg", width=100, height=100)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(50, 50)],  # Only one point
            attributes={}
        ))
        
        issues = list(_validate_image_annotations(img))
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("fewer than 2 points" in i.message for i in errors)
    
    def test_polyline_too_short(self):
        """Polyline shorter than threshold triggers warning."""
        img = ImageAnnotations(id=1, name="short.jpg", width=100, height=100)
        # Very short line (length ~1.4px)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(50, 50), Point(51, 51)],
            attributes={}
        ))
        
        issues = list(_validate_image_annotations(img))
        warnings = [i for i in issues if i.level == "warning"]
        assert len(warnings) >= 1
        assert any("below threshold" in i.message for i in warnings)
    
    def test_polygon_too_few_points(self):
        """Polygon with < 3 points triggers error."""
        img = ImageAnnotations(id=1, name="bad.jpg", width=100, height=100)
        img.polygons.append(PolygonAnnotation(
            label="ear_outer_contour",
            points=[Point(0, 0), Point(10, 10)]  # Only two points
        ))
        
        issues = list(_validate_image_annotations(img))
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("fewer than 3 points" in i.message for i in errors)
    
    def test_polygon_too_small(self):
        """Polygon with tiny area triggers warning."""
        img = ImageAnnotations(id=1, name="tiny.jpg", width=100, height=100)
        # Very small triangle
        img.polygons.append(PolygonAnnotation(
            label="franks_sign_region",
            points=[Point(50, 50), Point(51, 50), Point(50, 51)]  # Area ~0.5
        ))
        
        issues = list(_validate_image_annotations(img))
        warnings = [i for i in issues if i.level == "warning"]
        assert any("below threshold" in i.message for i in warnings)
    
    def test_self_intersecting_polygon_warning(self):
        """Self-intersecting polygon triggers warning."""
        img = ImageAnnotations(id=1, name="intersect.jpg", width=100, height=100)
        # Figure-8 / bowtie shape
        img.polygons.append(PolygonAnnotation(
            label="ear_outer_contour",
            points=[Point(0, 0), Point(100, 100), Point(100, 0), Point(0, 100)]
        ))
        
        issues = list(_validate_image_annotations(img))
        warnings = [i for i in issues if i.level == "warning"]
        assert any("self-intersecting" in i.message for i in warnings)


# ============================================================
# CLINICAL SCHEMA TESTS
# ============================================================

class TestClinicalSchema:
    """Tests for ClinicalSchema Pandera model."""
    
    def test_valid_data_passes(self):
        """Valid clinical data passes validation."""
        import pandas as pd
        
        data = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "fs_right": [1, 0],
            "fs_left": [0, 1],
            "gender": ["M", "F"],
            "age": [45, 60],
            "hdl": [50.0, 45.0],
            "ldl": [120.0, 140.0],
            "total_cholesterol": [200.0, 220.0],
            "triglycerides": [150.0, 180.0],
            "ef": [55.0, 50.0],
            "syntax_score": [10.0, 15.0],
        })
        
        validated = ClinicalSchema.validate(data)
        assert len(validated) == 2
    
    def test_missing_patient_id_fails(self):
        """Missing patient_id fails validation."""
        import pandas as pd
        
        data = pd.DataFrame({
            "patient_id": [None, "P002"],
            "fs_right": [1, 0],
            "fs_left": [0, 1],
            "gender": ["M", "F"],
            "age": [45, 60],
            "hdl": [50.0, 45.0],
            "ldl": [120.0, 140.0],
            "total_cholesterol": [200.0, 220.0],
            "triglycerides": [150.0, 180.0],
            "ef": [55.0, 50.0],
            "syntax_score": [10.0, 15.0],
        })
        
        with pytest.raises(Exception):  # SchemaError
            ClinicalSchema.validate(data)
    
    def test_invalid_age_fails(self):
        """Age outside 0-120 fails validation."""
        import pandas as pd
        
        data = pd.DataFrame({
            "patient_id": ["P001"],
            "fs_right": [1],
            "fs_left": [0],
            "gender": ["M"],
            "age": [150],  # Invalid
            "hdl": [50.0],
            "ldl": [120.0],
            "total_cholesterol": [200.0],
            "triglycerides": [150.0],
            "ef": [55.0],
            "syntax_score": [10.0],
        })
        
        with pytest.raises(Exception):
            ClinicalSchema.validate(data)
    
    def test_nullable_fields_accept_none(self):
        """Nullable fields accept None values for float/string types."""
        import pandas as pd
        
        # Integer columns (fs_right, fs_left, age) can't be None with coerce=True
        # Test that float/string nullable fields can be None
        data = pd.DataFrame({
            "patient_id": ["P001"],
            "fs_right": [1],
            "fs_left": [0],
            "gender": [None],  # String nullable
            "age": [45],       # Int - can't be None with coercion
            "hdl": [None],     # Float nullable
            "ldl": [None],
            "total_cholesterol": [None],
            "triglycerides": [None],
            "ef": [None],
            "syntax_score": [None],
        })
        
        validated = ClinicalSchema.validate(data)
        assert len(validated) == 1


# ============================================================
# CONSTANTS TESTS
# ============================================================

class TestConstants:
    """Tests for module constants."""
    
    def test_required_labels_subset_of_expected(self):
        """Required labels are subset of expected labels."""
        assert REQUIRED_LABELS.issubset(EXPECTED_LABELS)
    
    def test_thresholds_positive(self):
        """Validation thresholds are positive."""
        assert MIN_POLYLINE_LENGTH_PX > 0
        assert MIN_POLYGON_AREA_PX2 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
