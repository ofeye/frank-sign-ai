"""Comprehensive tests for geometric feature extraction module.

This test suite covers:
- Geometry utility functions (arc length, curvature, area, etc.)
- Feature extractor class
- Edge cases and error handling
"""

import math
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.cvat_parser import (
    Point,
    PointAnnotation,
    PolylineAnnotation,
    PolygonAnnotation,
    ImageAnnotations,
)
from franksign.data.geometric_features import (
    # Utility functions
    calculate_arc_length,
    calculate_euclidean_distance,
    calculate_discrete_curvature,
    calculate_polygon_area,
    calculate_centroid,
    calculate_bounding_box,
    # Feature classes
    FrankSignLineFeatures,
    FrankSignRegionFeatures,
    EarContourFeatures,
    LocalizationFeatures,
    ImageFeatures,
    # Main extractor
    GeometricFeatureExtractor,
    extract_features_batch,
    features_to_dataframe,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_image():
    """Image with basic Frank Sign annotation."""
    img = ImageAnnotations(id=1, name="simple.jpg", width=200, height=200)
    img.polylines.append(PolylineAnnotation(
        label="franks_sign_line",
        points=[Point(10, 50), Point(50, 50), Point(90, 50)],
        attributes={"presence": "present", "depth": "moderate"}
    ))
    return img


@pytest.fixture
def complete_image():
    """Image with all annotation types."""
    img = ImageAnnotations(id=2, name="complete.jpg", width=300, height=400)
    
    # Ear contour (rectangle for easy calculations)
    img.polygons.append(PolygonAnnotation(
        label="ear_outer_contour",
        points=[Point(50, 50), Point(250, 50), Point(250, 350), Point(50, 350)]
    ))
    
    # Frank Sign line (curved)
    img.polylines.append(PolylineAnnotation(
        label="franks_sign_line",
        points=[Point(100, 200), Point(150, 180), Point(200, 200)],
        attributes={"presence": "present", "depth": "deep"}
    ))
    
    # Frank Sign region
    img.polygons.append(PolygonAnnotation(
        label="franks_sign_region",
        points=[Point(100, 190), Point(200, 190), Point(200, 210), Point(100, 210)]
    ))
    
    return img


@pytest.fixture
def empty_image():
    """Image without any annotations."""
    return ImageAnnotations(id=3, name="empty.jpg", width=100, height=100)


# ============================================================
# UTILITY FUNCTION TESTS
# ============================================================

class TestCalculateArcLength:
    """Tests for arc length calculation."""
    
    def test_straight_horizontal_line(self):
        """Horizontal line arc length."""
        points = np.array([[0, 0], [10, 0], [20, 0]])
        assert calculate_arc_length(points) == pytest.approx(20.0)
    
    def test_straight_vertical_line(self):
        """Vertical line arc length."""
        points = np.array([[5, 0], [5, 15], [5, 25]])
        assert calculate_arc_length(points) == pytest.approx(25.0)
    
    def test_diagonal_line(self):
        """3-4-5 triangle diagonal."""
        points = np.array([[0, 0], [3, 4]])
        assert calculate_arc_length(points) == pytest.approx(5.0)
    
    def test_multi_segment(self):
        """Multiple segments sum correctly."""
        # 5 + 3 = 8 total
        points = np.array([[0, 0], [3, 4], [6, 4]])
        assert calculate_arc_length(points) == pytest.approx(8.0)
    
    def test_single_point_returns_zero(self):
        """Single point has zero length."""
        points = np.array([[42, 42]])
        assert calculate_arc_length(points) == 0.0
    
    def test_empty_array_returns_zero(self):
        """Empty array has zero length."""
        points = np.array([]).reshape(0, 2)
        assert calculate_arc_length(points) == 0.0


class TestCalculateEuclideanDistance:
    """Tests for Euclidean distance."""
    
    def test_horizontal_distance(self):
        """Simple horizontal distance."""
        p1, p2 = np.array([0, 0]), np.array([10, 0])
        assert calculate_euclidean_distance(p1, p2) == pytest.approx(10.0)
    
    def test_vertical_distance(self):
        """Simple vertical distance."""
        p1, p2 = np.array([0, 0]), np.array([0, 10])
        assert calculate_euclidean_distance(p1, p2) == pytest.approx(10.0)
    
    def test_diagonal_distance(self):
        """3-4-5 triangle."""
        p1, p2 = np.array([0, 0]), np.array([3, 4])
        assert calculate_euclidean_distance(p1, p2) == pytest.approx(5.0)
    
    def test_same_point_is_zero(self):
        """Distance from point to itself is zero."""
        p = np.array([5, 5])
        assert calculate_euclidean_distance(p, p) == pytest.approx(0.0)
    
    def test_negative_coordinates(self):
        """Works with negative coordinates."""
        p1, p2 = np.array([-3, -4]), np.array([0, 0])
        assert calculate_euclidean_distance(p1, p2) == pytest.approx(5.0)


class TestCalculateDiscreteCurvature:
    """Tests for discrete curvature calculation."""
    
    def test_straight_line_zero_curvature(self):
        """Straight line has near-zero curvature everywhere."""
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        curvatures = calculate_discrete_curvature(points)
        assert len(curvatures) == 3  # N-2 interior points
        # Due to floating point precision, near-zero is acceptable
        assert np.allclose(curvatures, 0.0, atol=1e-3)
    
    def test_right_angle_has_curvature(self):
        """90-degree turn has non-zero curvature."""
        points = np.array([[0, 0], [1, 0], [1, 1]])
        curvatures = calculate_discrete_curvature(points)
        assert len(curvatures) == 1
        assert curvatures[0] > 0  # Should have curvature
    
    def test_two_points_returns_empty(self):
        """Less than 3 points can't have curvature."""
        points = np.array([[0, 0], [1, 1]])
        curvatures = calculate_discrete_curvature(points)
        assert len(curvatures) == 0
    
    def test_three_points_returns_one_curvature(self):
        """Exactly 3 points gives 1 curvature value."""
        points = np.array([[0, 0], [1, 1], [2, 0]])
        curvatures = calculate_discrete_curvature(points)
        assert len(curvatures) == 1
    
    def test_semicircle_has_consistent_curvature(self):
        """Points on semicircle should have similar curvature."""
        # Create semicircle points
        angles = np.linspace(0, np.pi, 10)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        curvatures = calculate_discrete_curvature(points)
        # All curvatures should be similar (within some tolerance)
        assert np.std(curvatures) < np.mean(curvatures) * 0.3


class TestCalculatePolygonArea:
    """Tests for polygon area calculation."""
    
    def test_unit_square(self):
        """1x1 square has area 1."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert calculate_polygon_area(square) == pytest.approx(1.0)
    
    def test_rectangle(self):
        """3x4 rectangle has area 12."""
        rect = np.array([[0, 0], [3, 0], [3, 4], [0, 4]])
        assert calculate_polygon_area(rect) == pytest.approx(12.0)
    
    def test_right_triangle(self):
        """Triangle with legs 3 and 4 has area 6."""
        triangle = np.array([[0, 0], [3, 0], [0, 4]])
        assert calculate_polygon_area(triangle) == pytest.approx(6.0)
    
    def test_reversed_winding_still_positive(self):
        """Area is always positive regardless of winding."""
        cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert calculate_polygon_area(cw_square) == pytest.approx(1.0)
    
    def test_two_points_returns_zero(self):
        """Less than 3 points returns zero area."""
        line = np.array([[0, 0], [1, 1]])
        assert calculate_polygon_area(line) == 0.0
    
    def test_degenerate_polygon_near_zero(self):
        """Collinear points form degenerate polygon with ~zero area."""
        line = np.array([[0, 0], [1, 0], [2, 0]])
        assert calculate_polygon_area(line) == pytest.approx(0.0, abs=1e-6)


class TestCalculateCentroid:
    """Tests for centroid calculation."""
    
    def test_square_centroid(self):
        """Centroid of square is at center."""
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        cx, cy = calculate_centroid(square)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(1.0)
    
    def test_single_point_centroid(self):
        """Centroid of single point is the point itself."""
        point = np.array([[5, 10]])
        cx, cy = calculate_centroid(point)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(10.0)
    
    def test_asymmetric_polygon(self):
        """Centroid of asymmetric shape."""
        # Triangle with vertices at (0,0), (6,0), (3,6)
        triangle = np.array([[0, 0], [6, 0], [3, 6]])
        cx, cy = calculate_centroid(triangle)
        assert cx == pytest.approx(3.0)
        assert cy == pytest.approx(2.0)


class TestCalculateBoundingBox:
    """Tests for bounding box calculation."""
    
    def test_unit_square_bbox(self):
        """Bounding box of unit square at origin."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        x, y, w, h = calculate_bounding_box(square)
        assert (x, y, w, h) == pytest.approx((0, 0, 1, 1))
    
    def test_offset_rectangle(self):
        """Bounding box of offset rectangle."""
        rect = np.array([[10, 20], [30, 20], [30, 50], [10, 50]])
        x, y, w, h = calculate_bounding_box(rect)
        assert x == pytest.approx(10)
        assert y == pytest.approx(20)
        assert w == pytest.approx(20)
        assert h == pytest.approx(30)
    
    def test_irregular_polygon(self):
        """Bounding box of irregular polygon."""
        poly = np.array([[5, 5], [15, 8], [12, 20], [3, 15]])
        x, y, w, h = calculate_bounding_box(poly)
        assert x == pytest.approx(3)
        assert y == pytest.approx(5)
        assert w == pytest.approx(12)  # 15-3
        assert h == pytest.approx(15)  # 20-5


# ============================================================
# FEATURE EXTRACTOR TESTS
# ============================================================

class TestGeometricFeatureExtractor:
    """Tests for GeometricFeatureExtractor class."""
    
    def test_extract_all_returns_image_features(self, simple_image):
        """extract_all returns ImageFeatures object."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        
        assert isinstance(features, ImageFeatures)
        assert features.image_name == "simple.jpg"
        assert features.image_id == 1
    
    def test_straight_line_tortuosity_one(self, simple_image):
        """Straight Frank Sign line has tortuosity ~= 1.0."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.tortuosity == pytest.approx(1.0, rel=1e-3)
    
    def test_curved_line_tortuosity_greater_than_one(self, complete_image):
        """Curved Frank Sign line has tortuosity > 1.0."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.tortuosity > 1.0
    
    def test_relative_length_calculated(self, complete_image):
        """Relative length computed when ear contour available."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.relative_length is not None
        assert 0 < features.frank_sign_line.relative_length < 1
    
    def test_no_ear_contour_no_relative_length(self, simple_image):
        """Relative length is None when ear contour missing."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.relative_length is None
    
    def test_empty_image_returns_none_features(self, empty_image):
        """Empty image returns ImageFeatures with None sub-features."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(empty_image)
        
        assert features.frank_sign_line is None
        assert features.frank_sign_region is None
        assert features.ear_contour is None
        assert features.has_frank_sign == False
    
    def test_region_features_extracted(self, complete_image):
        """Region polygon features are extracted."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        
        assert features.frank_sign_region is not None
        assert features.frank_sign_region.area > 0
        assert features.frank_sign_region.perimeter > 0
        assert 0 < features.frank_sign_region.compactness <= 1
    
    def test_ear_contour_features(self, complete_image):
        """Ear contour features are extracted."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        
        assert features.ear_contour is not None
        # Our fixture has 200x300 rectangle ear contour
        assert features.ear_contour.width == pytest.approx(200.0)
        assert features.ear_contour.height == pytest.approx(300.0)
        assert features.ear_contour.area == pytest.approx(60000.0)
    
    def test_localization_features(self, complete_image):
        """Localization features are extracted."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        
        assert features.localization is not None
        # Relative position should be normalized by ear height
        assert -1 < features.localization.relative_x < 1
        assert -1 < features.localization.relative_y < 1
    
    def test_scale_factor_applied(self, simple_image):
        """Scale factor converts pixels to mm."""
        # 10 pixels per mm
        extractor = GeometricFeatureExtractor(scale_factor=10.0)
        features = extractor.extract_all(simple_image)
        
        # Line is 80px long, should be 8mm with scale factor
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.length == pytest.approx(8.0)
    
    def test_attributes_preserved(self, simple_image):
        """Frank Sign attributes are preserved."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        
        assert "presence" in features.frank_sign_attributes
        assert features.frank_sign_attributes["presence"] == "present"
        assert "depth" in features.frank_sign_attributes


class TestImageFeaturesToDict:
    """Tests for ImageFeatures.to_dict() method."""
    
    def test_basic_fields_present(self, simple_image):
        """Basic fields are in dictionary."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        d = features.to_dict()
        
        assert "image_name" in d
        assert "image_id" in d
        assert "has_frank_sign" in d
        assert d["image_name"] == "simple.jpg"
    
    def test_frank_line_fields_present(self, complete_image):
        """Frank Sign line fields are in dictionary."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(complete_image)
        d = features.to_dict()
        
        assert "fs_length" in d
        assert "fs_tortuosity" in d
        assert "fs_curvature_mean" in d
        assert "fs_relative_length" in d
    
    def test_attributes_merged_into_dict(self, simple_image):
        """Attributes are merged into flat dictionary."""
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(simple_image)
        d = features.to_dict()
        
        assert "presence" in d or "depth" in d


# ============================================================
# BATCH AND DATAFRAME TESTS
# ============================================================

class TestExtractFeaturesBatch:
    """Tests for batch feature extraction."""
    
    def test_batch_returns_list(self, simple_image, complete_image):
        """Batch extraction returns list of same length."""
        images = [simple_image, complete_image]
        features = extract_features_batch(images)
        
        assert len(features) == 2
        assert all(isinstance(f, ImageFeatures) for f in features)
    
    def test_batch_with_scale_factor(self, simple_image):
        """Scale factor is applied to all images in batch."""
        features = extract_features_batch([simple_image], scale_factor=10.0)
        
        assert features[0].frank_sign_line.length == pytest.approx(8.0)


class TestFeaturesToDataFrame:
    """Tests for DataFrame conversion."""
    
    def test_creates_dataframe(self, simple_image, complete_image):
        """Creates pandas DataFrame."""
        features = extract_features_batch([simple_image, complete_image])
        df = features_to_dataframe(features)
        
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_dataframe_columns(self, complete_image):
        """DataFrame has expected columns."""
        features = extract_features_batch([complete_image])
        df = features_to_dataframe(features)
        
        expected_cols = ["image_name", "image_id", "has_frank_sign", 
                         "fs_length", "fs_tortuosity"]
        for col in expected_cols:
            assert col in df.columns


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================

class TestEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_single_point_polyline(self):
        """Polyline with single point returns None features."""
        img = ImageAnnotations(id=1, name="edge.jpg", width=100, height=100)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(50, 50)],
            attributes={}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_line is None
    
    def test_two_point_line_valid(self):
        """Two-point polyline is valid."""
        img = ImageAnnotations(id=1, name="edge.jpg", width=100, height=100)
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(0, 0), Point(10, 10)],
            attributes={"presence": "present"}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_line is not None
        assert features.frank_sign_line.num_points == 2
    
    def test_very_small_polygon(self):
        """Very small polygon (nearly degenerate)."""
        img = ImageAnnotations(id=1, name="edge.jpg", width=100, height=100)
        # Triangle with very small area
        img.polygons.append(PolygonAnnotation(
            label="franks_sign_region",
            points=[Point(50, 50), Point(50.001, 50), Point(50, 50.001)]
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        assert features.frank_sign_region is not None
        assert features.frank_sign_region.area >= 0
    
    def test_coincident_start_end_points(self):
        """Polyline with same start and end point."""
        img = ImageAnnotations(id=1, name="edge.jpg", width=100, height=100)
        # Closed loop
        img.polylines.append(PolylineAnnotation(
            label="franks_sign_line",
            points=[Point(0, 0), Point(10, 10), Point(0, 0)],
            attributes={"presence": "present"}
        ))
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract_all(img)
        
        # Should handle without error
        assert features.frank_sign_line is not None
        # Euclidean distance is 0, tortuosity should be handled
        assert features.frank_sign_line.euclidean_distance == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
