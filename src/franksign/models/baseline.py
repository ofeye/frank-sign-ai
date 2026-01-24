"""Classical computer vision baseline for Frank Sign detection.

This module provides a Canny edge detection baseline that serves as:
1. A reference point for deep learning model comparison
2. An interpretable, explainable approach for clinical validation
3. A fallback when training data is limited

Example:
    >>> from franksign.models.baseline import CannyBaseline
    >>> baseline = CannyBaseline()
    >>> edges, contours = baseline.detect(image)
    >>> features = baseline.extract_features(contours)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class ContourFeatures:
    """Geometric features extracted from a contour."""
    length: float
    area: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    orientation: float  # degrees
    curvature_mean: float
    curvature_std: float


class CannyBaseline:
    """Canny edge detection baseline for Frank Sign segmentation.
    
    This classical CV approach provides:
    - Edge detection using Canny algorithm
    - Contour extraction and filtering
    - Geometric feature computation
    - Earlobe region focusing using morphological operations
    
    Attributes:
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
        blur_kernel: Gaussian blur kernel size
        min_contour_length: Minimum contour length to consider
    
    Example:
        >>> baseline = CannyBaseline(low_threshold=50, high_threshold=150)
        >>> edges = baseline.detect_edges(image)
        >>> mask = baseline.create_mask(edges, image.shape[:2])
    """
    
    def __init__(
        self,
        low_threshold: int = 50,
        high_threshold: int = 150,
        blur_kernel: int = 5,
        min_contour_length: float = 20.0,
    ):
        """Initialize Canny baseline.
        
        Args:
            low_threshold: Lower Canny threshold
            high_threshold: Upper Canny threshold
            blur_kernel: Size of Gaussian blur kernel (must be odd)
            min_contour_length: Minimum arc length to keep contour
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for CannyBaseline")
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.blur_kernel = blur_kernel
        self.min_contour_length = min_contour_length
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny algorithm.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Binary edge image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return edges
    
    def extract_contours(
        self, 
        edges: np.ndarray,
        filter_by_length: bool = True,
    ) -> List[np.ndarray]:
        """Extract contours from edge image.
        
        Args:
            edges: Binary edge image from detect_edges
            filter_by_length: Filter out short contours
        
        Returns:
            List of contours (each is Nx1x2 array)
        """
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if filter_by_length:
            contours = [
                c for c in contours 
                if cv2.arcLength(c, closed=False) >= self.min_contour_length
            ]
        
        return contours
    
    def filter_diagonal_contours(
        self,
        contours: List[np.ndarray],
        angle_range: Tuple[float, float] = (20, 70),
    ) -> List[np.ndarray]:
        """Filter contours to keep only diagonal ones (likely Frank Sign).
        
        Frank Sign typically runs diagonally across the earlobe.
        
        Args:
            contours: List of contours
            angle_range: (min_angle, max_angle) in degrees from horizontal
        
        Returns:
            Filtered list of diagonal contours
        """
        diagonal_contours = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
            
            # Fit line to contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate angle from horizontal
            angle = abs(np.degrees(np.arctan2(vy, vx)))
            if angle > 90:
                angle = 180 - angle
            
            # Keep if within diagonal range
            if angle_range[0] <= angle <= angle_range[1]:
                diagonal_contours.append(contour)
        
        return diagonal_contours
    
    def create_mask(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple[int, int],
        thickness: int = 3,
    ) -> np.ndarray:
        """Create binary mask from contours.
        
        Args:
            contours: List of contours to draw
            image_shape: (height, width) of output mask
            thickness: Line thickness
        
        Returns:
            Binary mask with contours drawn
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 1, thickness)
        return mask
    
    def extract_features(self, contour: np.ndarray) -> ContourFeatures:
        """Extract geometric features from a single contour.
        
        Args:
            contour: Contour array of shape (N, 1, 2)
        
        Returns:
            ContourFeatures dataclass
        """
        # Arc length
        length = cv2.arcLength(contour, closed=False)
        
        # Area (for closed contours or convex hull)
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = contour.mean(axis=0)[0]
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Orientation (from fitted line)
        if len(contour) >= 5:
            [vx, vy, _, _] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            orientation = float(np.degrees(np.arctan2(vy, vx)))
        else:
            orientation = 0.0
        
        # Curvature (approximate from angle changes)
        points = contour.reshape(-1, 2)
        curvatures = self._compute_curvature(points)
        
        return ContourFeatures(
            length=length,
            area=area,
            centroid=(cx, cy),
            bounding_box=(x, y, w, h),
            orientation=orientation,
            curvature_mean=float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0,
            curvature_std=float(np.std(curvatures)) if len(curvatures) > 0 else 0.0,
        )
    
    def _compute_curvature(self, points: np.ndarray) -> np.ndarray:
        """Compute discrete curvature at each interior point."""
        if len(points) < 3:
            return np.array([])
        
        curvatures = []
        for i in range(1, len(points) - 1):
            p_prev = points[i - 1]
            p_curr = points[i]
            p_next = points[i + 1]
            
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            # Chord length
            chord = np.linalg.norm(p_next - p_prev)
            
            # Curvature approximation
            if chord > 1e-8:
                curvature = 2 * np.sin(np.pi - angle) / chord
            else:
                curvature = 0.0
            
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def detect(
        self, 
        image: np.ndarray,
        filter_diagonal: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Full detection pipeline.
        
        Args:
            image: Input image
            filter_diagonal: Whether to filter for diagonal contours
        
        Returns:
            Tuple of (edge_image, filtered_contours)
        """
        edges = self.detect_edges(image)
        contours = self.extract_contours(edges)
        
        if filter_diagonal:
            contours = self.filter_diagonal_contours(contours)
        
        return edges, contours
    
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Predict Frank Sign mask for a single image.
        
        Compatible interface with deep learning models.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Binary mask of predicted Frank Sign location
        """
        _, contours = self.detect(image, filter_diagonal=True)
        
        if not contours:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        return self.create_mask(contours, image.shape[:2])


if __name__ == "__main__":
    print("Canny Baseline module loaded.")
    print(f"OpenCV available: {HAS_OPENCV}")
