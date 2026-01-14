"""Geometric feature extraction from Frank Sign annotations.

This module extracts quantitative geometric features from polyline and polygon
annotations, including:
- Arc length and area
- Curvature analysis
- Tortuosity
- Normalized localization

Example:
    >>> from franksign.data import CVATParser, GeometricFeatureExtractor
    >>> parser = CVATParser("annotations.xml")
    >>> project = parser.parse()
    >>> extractor = GeometricFeatureExtractor()
    >>> 
    >>> for image in project.images:
    ...     features = extractor.extract_all(image)
    ...     print(features)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from franksign.data.cvat_parser import (
    ImageAnnotations,
    Point,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)


# ============================================================
# FEATURE DATA CLASSES
# ============================================================

@dataclass
class FrankSignLineFeatures:
    """Geometric features extracted from Frank Sign line (polyline).
    
    All length measurements are in pixels unless a scale factor is provided.
    """
    length: float  # Arc length
    euclidean_distance: float  # Start to end straight line
    tortuosity: float  # length / euclidean_distance (≥1.0)
    curvature_mean: float  # Mean discrete curvature
    curvature_max: float  # Maximum local curvature
    curvature_std: float  # Standard deviation of curvature
    num_points: int  # Number of annotation points
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    centroid: Tuple[float, float]
    
    # Normalized versions (if ear contour available)
    relative_length: Optional[float] = None  # Relative to ear height


@dataclass
class FrankSignRegionFeatures:
    """Geometric features from Frank Sign region polygon."""
    area: float  # Polygon area in pixels²
    perimeter: float  # Polygon perimeter
    compactness: float  # 4π × area / perimeter²
    centroid: Tuple[float, float]
    bounding_box: Tuple[float, float, float, float]  # x, y, width, height


@dataclass
class EarContourFeatures:
    """Geometric features from ear outer contour."""
    area: float
    perimeter: float
    height: float  # Vertical extent
    width: float  # Horizontal extent
    aspect_ratio: float  # height / width
    centroid: Tuple[float, float]


@dataclass
class LocalizationFeatures:
    """Normalized position of Frank Sign relative to ear anatomy."""
    # Position relative to ear centroid (normalized by ear height)
    relative_x: float  # Positive = towards face
    relative_y: float  # Positive = towards bottom
    
    # Distance ratios
    distance_to_earlobe_tip: Optional[float] = None
    distance_to_tragus: Optional[float] = None
    
    # Angle (from ear canal center)
    angle_from_center: Optional[float] = None  # Radians


@dataclass
class ImageFeatures:
    """Complete feature set for one image."""
    image_name: str
    image_id: int
    has_frank_sign: bool
    
    # Primary features
    frank_sign_line: Optional[FrankSignLineFeatures] = None
    frank_sign_region: Optional[FrankSignRegionFeatures] = None
    ear_contour: Optional[EarContourFeatures] = None
    localization: Optional[LocalizationFeatures] = None
    
    # Metadata from annotations
    frank_sign_attributes: Dict[str, str] = field(default_factory=dict)
    image_quality: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for DataFrame export."""
        result = {
            "image_name": self.image_name,
            "image_id": self.image_id,
            "has_frank_sign": self.has_frank_sign,
        }
        
        if self.frank_sign_line:
            result.update({
                "fs_length": self.frank_sign_line.length,
                "fs_euclidean": self.frank_sign_line.euclidean_distance,
                "fs_tortuosity": self.frank_sign_line.tortuosity,
                "fs_curvature_mean": self.frank_sign_line.curvature_mean,
                "fs_curvature_max": self.frank_sign_line.curvature_max,
                "fs_curvature_std": self.frank_sign_line.curvature_std,
                "fs_num_points": self.frank_sign_line.num_points,
                "fs_relative_length": self.frank_sign_line.relative_length,
            })
        
        if self.frank_sign_region:
            result.update({
                "fs_region_area": self.frank_sign_region.area,
                "fs_region_perimeter": self.frank_sign_region.perimeter,
                "fs_region_compactness": self.frank_sign_region.compactness,
            })
        
        if self.ear_contour:
            result.update({
                "ear_area": self.ear_contour.area,
                "ear_height": self.ear_contour.height,
                "ear_width": self.ear_contour.width,
                "ear_aspect_ratio": self.ear_contour.aspect_ratio,
            })
        
        if self.localization:
            result.update({
                "loc_relative_x": self.localization.relative_x,
                "loc_relative_y": self.localization.relative_y,
            })
        
        # Add categorical attributes
        result.update(self.frank_sign_attributes)
        result.update(self.image_quality)
        
        return result


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def calculate_arc_length(points: np.ndarray) -> float:
    """Calculate total arc length of a polyline.
    
    Args:
        points: Array of shape (N, 2) with x,y coordinates.
        
    Returns:
        Sum of segment lengths.
    """
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def calculate_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(p2 - p1))


def calculate_discrete_curvature(points: np.ndarray) -> np.ndarray:
    """Calculate discrete curvature at each interior point.
    
    Uses the inscribed circle radius formula:
    κ = 2 * sin(θ) / |p_{i+1} - p_{i-1}|
    
    where θ is the angle at point i.
    
    Args:
        points: Array of shape (N, 2).
        
    Returns:
        Array of curvature values of shape (N-2,).
    """
    if len(points) < 3:
        return np.array([])
    
    curvatures = []
    for i in range(1, len(points) - 1):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Calculate chord length
        chord = np.linalg.norm(p_next - p_prev)
        
        # Curvature approximation
        if chord > 1e-8:
            curvature = 2 * np.sin(np.pi - angle) / chord
        else:
            curvature = 0.0
        
        curvatures.append(curvature)
    
    return np.array(curvatures)


def calculate_polygon_area(points: np.ndarray) -> float:
    """Calculate polygon area using shoelace formula.
    
    Args:
        points: Array of shape (N, 2) representing closed polygon.
        
    Returns:
        Absolute area (always positive).
    """
    if len(points) < 3:
        return 0.0
    
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    
    return abs(area) / 2.0


def calculate_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid of a set of points."""
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def calculate_bounding_box(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate bounding box (x, y, width, height)."""
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)


# ============================================================
# FEATURE EXTRACTOR CLASS
# ============================================================

class GeometricFeatureExtractor:
    """Extract geometric features from annotated ear images.
    
    Attributes:
        scale_factor: Pixels per mm (if ruler detected).
        
    Example:
        >>> extractor = GeometricFeatureExtractor()
        >>> features = extractor.extract_all(image_annotations)
        >>> print(f"Frank Sign length: {features.frank_sign_line.length:.2f} px")
    """
    
    def __init__(self, scale_factor: Optional[float] = None):
        """Initialize feature extractor.
        
        Args:
            scale_factor: Pixels per millimeter for dimensional calibration.
                If provided, length measurements will be converted to mm.
        """
        self.scale_factor = scale_factor
    
    def extract_all(self, image: ImageAnnotations) -> ImageFeatures:
        """Extract all features from an image.
        
        Args:
            image: Parsed image annotations.
            
        Returns:
            Complete ImageFeatures object.
        """
        # Get ear contour first (needed for normalization)
        ear_contour = self._extract_ear_contour(image)
        
        # Get Frank Sign features
        frank_line = self._extract_frank_sign_line(image, ear_contour)
        frank_region = self._extract_frank_sign_region(image)
        
        # Get localization
        localization = self._extract_localization(image, ear_contour)
        
        # Get attributes
        frank_attrs = {}
        for polyline in image.polylines:
            if polyline.label == "franks_sign_line":
                frank_attrs = polyline.attributes.copy()
                break
        
        quality_attrs = {}
        for point in image.points:
            if point.label == "image_quality_assessment":
                quality_attrs = point.attributes.copy()
                break
        
        return ImageFeatures(
            image_name=image.name,
            image_id=image.id,
            has_frank_sign=image.has_frank_sign,
            frank_sign_line=frank_line,
            frank_sign_region=frank_region,
            ear_contour=ear_contour,
            localization=localization,
            frank_sign_attributes=frank_attrs,
            image_quality=quality_attrs,
        )
    
    def _extract_frank_sign_line(
        self, 
        image: ImageAnnotations,
        ear_contour: Optional[EarContourFeatures] = None
    ) -> Optional[FrankSignLineFeatures]:
        """Extract features from Frank Sign polyline."""
        # Find Frank Sign line annotation
        frank_line = None
        for polyline in image.polylines:
            if polyline.label == "franks_sign_line":
                frank_line = polyline
                break
        
        if frank_line is None or len(frank_line.points) < 2:
            return None
        
        points = frank_line.to_array()
        
        # Basic measurements
        arc_length = calculate_arc_length(points)
        start = points[0]
        end = points[-1]
        euclidean = calculate_euclidean_distance(start, end)
        
        # Tortuosity
        tortuosity = arc_length / euclidean if euclidean > 1e-8 else 1.0
        
        # Curvature
        curvatures = calculate_discrete_curvature(points)
        if len(curvatures) > 0:
            curvature_mean = float(np.mean(curvatures))
            curvature_max = float(np.max(curvatures))
            curvature_std = float(np.std(curvatures))
        else:
            curvature_mean = curvature_max = curvature_std = 0.0
        
        # Centroid
        centroid = calculate_centroid(points)
        
        # Relative length
        relative_length = None
        if ear_contour and ear_contour.height > 0:
            relative_length = arc_length / ear_contour.height
        
        # Apply scale factor if available
        if self.scale_factor:
            arc_length /= self.scale_factor
            euclidean /= self.scale_factor
        
        return FrankSignLineFeatures(
            length=arc_length,
            euclidean_distance=euclidean,
            tortuosity=tortuosity,
            curvature_mean=curvature_mean,
            curvature_max=curvature_max,
            curvature_std=curvature_std,
            num_points=len(points),
            start_point=tuple(start),
            end_point=tuple(end),
            centroid=centroid,
            relative_length=relative_length,
        )
    
    def _extract_frank_sign_region(self, image: ImageAnnotations) -> Optional[FrankSignRegionFeatures]:
        """Extract features from Frank Sign region polygon."""
        region = None
        for polygon in image.polygons:
            if polygon.label == "franks_sign_region":
                region = polygon
                break
        
        if region is None or len(region.points) < 3:
            return None
        
        points = region.to_array()
        
        area = calculate_polygon_area(points)
        perimeter = calculate_arc_length(np.vstack([points, points[0]]))  # Close polygon
        compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        centroid = calculate_centroid(points)
        bbox = calculate_bounding_box(points)
        
        return FrankSignRegionFeatures(
            area=area,
            perimeter=perimeter,
            compactness=compactness,
            centroid=centroid,
            bounding_box=bbox,
        )
    
    def _extract_ear_contour(self, image: ImageAnnotations) -> Optional[EarContourFeatures]:
        """Extract features from ear outer contour."""
        contour = None
        for polygon in image.polygons:
            if polygon.label == "ear_outer_contour":
                contour = polygon
                break
        
        if contour is None or len(contour.points) < 3:
            return None
        
        points = contour.to_array()
        
        area = calculate_polygon_area(points)
        perimeter = calculate_arc_length(np.vstack([points, points[0]]))
        bbox = calculate_bounding_box(points)
        centroid = calculate_centroid(points)
        
        x, y, width, height = bbox
        aspect_ratio = height / width if width > 0 else 1.0
        
        return EarContourFeatures(
            area=area,
            perimeter=perimeter,
            height=height,
            width=width,
            aspect_ratio=aspect_ratio,
            centroid=centroid,
        )
    
    def _extract_localization(
        self,
        image: ImageAnnotations,
        ear_contour: Optional[EarContourFeatures] = None
    ) -> Optional[LocalizationFeatures]:
        """Extract normalized localization features."""
        # Need Frank Sign line and ear contour
        frank_line = None
        for polyline in image.polylines:
            if polyline.label == "franks_sign_line":
                frank_line = polyline
                break
        
        if frank_line is None or ear_contour is None:
            return None
        
        # Get Frank Sign centroid
        fs_points = frank_line.to_array()
        fs_centroid = calculate_centroid(fs_points)
        
        # Normalize relative to ear centroid and height
        ear_cx, ear_cy = ear_contour.centroid
        rel_x = (fs_centroid[0] - ear_cx) / ear_contour.height
        rel_y = (fs_centroid[1] - ear_cy) / ear_contour.height
        
        return LocalizationFeatures(
            relative_x=rel_x,
            relative_y=rel_y,
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def extract_features_batch(
    images: List[ImageAnnotations],
    scale_factor: Optional[float] = None
) -> List[ImageFeatures]:
    """Extract features from multiple images.
    
    Args:
        images: List of image annotations.
        scale_factor: Optional pixels-per-mm scale.
        
    Returns:
        List of ImageFeatures for each image.
    """
    extractor = GeometricFeatureExtractor(scale_factor)
    return [extractor.extract_all(img) for img in images]


def features_to_dataframe(features: List[ImageFeatures]):
    """Convert list of features to pandas DataFrame.
    
    Args:
        features: List of ImageFeatures objects.
        
    Returns:
        pandas DataFrame with one row per image.
    """
    import pandas as pd
    
    records = [f.to_dict() for f in features]
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick test with sample data
    print("Geometric Feature Extractor module loaded.")
    print("Use with CVATParser to extract features from annotations.")
