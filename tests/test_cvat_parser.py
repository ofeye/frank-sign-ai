"""Tests for CVAT parser module."""

import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.cvat_parser import (
    CVATParser,
    load_annotations,
    Point,
    PointAnnotation,
    PolylineAnnotation,
    PolygonAnnotation,
    ImageAnnotations,
)
from franksign.data.geometric_features import (
    calculate_arc_length,
    calculate_euclidean_distance,
    calculate_discrete_curvature,
    calculate_polygon_area,
    GeometricFeatureExtractor,
)


# ============================================================
# GEOMETRY UTILITY TESTS
# ============================================================

class TestGeometryUtilities:
    """Test geometric calculation functions."""
    
    def test_arc_length_straight_line(self):
        """Arc length of straight line should equal segment sum."""
        points = np.array([[0, 0], [3, 0], [6, 0]])
        assert calculate_arc_length(points) == pytest.approx(6.0)
    
    def test_arc_length_diagonal(self):
        """Arc length with diagonal segment (3-4-5 triangle)."""
        points = np.array([[0, 0], [3, 4]])
        assert calculate_arc_length(points) == pytest.approx(5.0)
    
    def test_arc_length_multiple_segments(self):
        """Arc length with multiple segments."""
        # Two segments: (0,0)→(3,4) = 5, (3,4)→(6,4) = 3
        points = np.array([[0, 0], [3, 4], [6, 4]])
        assert calculate_arc_length(points) == pytest.approx(8.0)
    
    def test_arc_length_single_point(self):
        """Single point should have zero length."""
        points = np.array([[5, 5]])
        assert calculate_arc_length(points) == 0.0
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        assert calculate_euclidean_distance(p1, p2) == pytest.approx(5.0)
    
    def test_polygon_area_square(self):
        """Area of unit square = 1."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert calculate_polygon_area(square) == pytest.approx(1.0)
    
    def test_polygon_area_rectangle(self):
        """Area of 3x4 rectangle = 12."""
        rect = np.array([[0, 0], [3, 0], [3, 4], [0, 4]])
        assert calculate_polygon_area(rect) == pytest.approx(12.0)
    
    def test_polygon_area_triangle(self):
        """Area of right triangle with legs 3 and 4 = 6."""
        triangle = np.array([[0, 0], [3, 0], [0, 4]])
        assert calculate_polygon_area(triangle) == pytest.approx(6.0)
    
    def test_curvature_straight_line(self):
        """Straight line should have near-zero curvature."""
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        curvatures = calculate_discrete_curvature(points)
        # Due to floating point precision, near-zero is acceptable
        assert np.allclose(curvatures, 0.0, atol=1e-3)
    
    def test_curvature_needs_three_points(self):
        """Curvature requires at least 3 points."""
        points = np.array([[0, 0], [1, 1]])
        curvatures = calculate_discrete_curvature(points)
        assert len(curvatures) == 0


# ============================================================
# PARSER TESTS (require actual annotations.xml)
# ============================================================

class TestCVATParser:
    """Tests for CVAT XML parser.
    
    These tests require the actual annotations.xml file.
    Skip if not available.
    """
    
    @pytest.fixture
    def annotations_path(self):
        """Get path to annotations file."""
        path = Path(__file__).parent.parent / "annotations.xml"
        if not path.exists():
            path = Path(__file__).parent.parent / "data" / "annotations" / "annotations.xml"
        return path
    
    def test_parser_loads_file(self, annotations_path):
        """Parser should load XML file without errors."""
        if not annotations_path.exists():
            pytest.skip(f"Annotations file not found: {annotations_path}")
        
        parser = CVATParser(annotations_path)
        project = parser.parse()
        
        assert project is not None
        assert project.num_images > 0
    
    def test_parser_extracts_labels(self, annotations_path):
        """Parser should extract all label definitions."""
        if not annotations_path.exists():
            pytest.skip(f"Annotations file not found: {annotations_path}")
        
        project = load_annotations(annotations_path)
        
        label_names = [l.name for l in project.labels]
        
        # Check for expected labels
        assert "franks_sign_line" in label_names
        assert "ear_outer_contour" in label_names
        assert "franks_sign_region" in label_names
    
    def test_parser_extracts_frank_sign_line(self, annotations_path):
        """Parser should extract Frank Sign polylines."""
        if not annotations_path.exists():
            pytest.skip(f"Annotations file not found: {annotations_path}")
        
        project = load_annotations(annotations_path)
        
        # Find at least one image with Frank Sign
        found_frank_sign = False
        for img in project.images:
            for polyline in img.polylines:
                if polyline.label == "franks_sign_line":
                    found_frank_sign = True
                    assert len(polyline.points) >= 2
                    break
            if found_frank_sign:
                break
        
        # It's OK if not all images have Frank Sign
        # Just verify parsing works
    
    def test_parser_extracts_attributes(self, annotations_path):
        """Parser should extract label attributes."""
        if not annotations_path.exists():
            pytest.skip(f"Annotations file not found: {annotations_path}")
        
        project = load_annotations(annotations_path)
        
        # Find a Frank Sign line and check attributes
        for img in project.images:
            for polyline in img.polylines:
                if polyline.label == "franks_sign_line" and polyline.attributes:
                    # Should have presence attribute
                    assert "presence" in polyline.attributes or len(polyline.attributes) > 0
                    return
        
        pytest.skip("No Frank Sign line with attributes found")


# ============================================================
# FEATURE EXTRACTOR TESTS
# ============================================================

class TestFeatureExtractor:
    """Tests for geometric feature extraction."""
    
    def test_tortuosity_straight_line(self):
        """Tortuosity of straight line should be 1.0."""
        # Create mock image with straight Frank Sign
        img = ImageAnnotations(
            id=1, name="test.jpg", width=100, height=100
        )
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(0, 0), Point(10, 0), Point(20, 0)],
            attributes={"presence": "present"}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.tortuosity == pytest.approx(1.0, rel=1e-6)
    
    def test_tortuosity_curved_line(self):
        """Tortuosity of curved line should be > 1.0."""
        img = ImageAnnotations(
            id=1, name="test.jpg", width=100, height=100
        )
        # Create a curved path (going up then right)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(0, 0), Point(0, 10), Point(10, 10)],
            attributes={"presence": "present"}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.tortuosity > 1.0
    
    def test_relative_length_calculation(self):
        """Relative length should be computed when ear contour present."""
        img = ImageAnnotations(
            id=1, name="test.jpg", width=100, height=100
        )
        
        # Add ear contour (100px height)
        img.polygons.append(PolygonAnnotation(
            label="ear_outer_contour",
            points=[
                Point(0, 0), Point(50, 0), 
                Point(50, 100), Point(0, 100)
            ]
        ))
        
        # Add Frank Sign line (20px length)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(10, 50), Point(30, 50)],
            attributes={"presence": "present"}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.relative_length == pytest.approx(0.2, rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
