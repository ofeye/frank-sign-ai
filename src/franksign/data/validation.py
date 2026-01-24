"""Validation utilities for clinical data and CVAT annotations.

- Clinical validation uses Pandera to enforce basic schema checks.
- CVAT validation performs lightweight structural checks (labels and geometry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandera as pa
from pandera.typing import DataFrame, Series

from franksign.data.cvat_parser import CVATProject, ImageAnnotations
from franksign.data.geometric_features import calculate_arc_length, calculate_polygon_area


# Expected labels from docs/data_schema.md
EXPECTED_LABELS = {
    "ear_outer_contour",
    "franks_sign_region",
    "franks_sign_line",
    "ear_canal_center",
    "tragus_point",
    "antitragus_point",
    "antitragus_end_point",
    "intertragic_notch",
    "earlobe_tip",
    "ear_top",
    "earlobe_attachment_point",
    "image_quality_assessment",
    "annotation_metadata",
    "patient_metadata",
}

REQUIRED_LABELS = {
    "ear_outer_contour",
    "franks_sign_line",
}

MIN_POLYLINE_LENGTH_PX = 5.0
MIN_POLYGON_AREA_PX2 = 10.0


class ClinicalSchema(pa.DataFrameModel):
    """Minimal clinical schema (extend as production data grows)."""

    patient_id: Series[str] = pa.Field(nullable=False)
    fs_right: Series[int] = pa.Field(nullable=True, isin=[0, 1, None])
    fs_left: Series[int] = pa.Field(nullable=True, isin=[0, 1, None])
    gender: Series[str] = pa.Field(nullable=True, isin=["M", "F", None])
    age: Series[int] = pa.Field(nullable=True, ge=0, le=120)
    hdl: Series[float] = pa.Field(nullable=True, ge=0)
    ldl: Series[float] = pa.Field(nullable=True, ge=0)
    total_cholesterol: Series[float] = pa.Field(nullable=True, ge=0)
    triglycerides: Series[float] = pa.Field(nullable=True, ge=0)
    ef: Series[float] = pa.Field(nullable=True, ge=0, le=100)
    syntax_score: Series[float] = pa.Field(nullable=True, ge=0)

    class Config:
        coerce = True


@dataclass
class ValidationIssue:
    level: str  # e.g., "warning" or "error"
    message: str


def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Check if two segments p1-p2 and p3-p4 intersect (excluding colinear overlap handling)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    return (o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0) or (o1 * o2 < 0 and o3 * o4 < 0)


def _is_self_intersecting(points) -> bool:
    """Detect self-intersections in a polygon represented as ndarray (N,2)."""
    pts = [tuple(p) for p in points]
    n = len(pts)
    if n < 4:
        return False
    edges = [
        (pts[i], pts[(i + 1) % n])
        for i in range(n)
    ]
    for i, (a1, a2) in enumerate(edges):
        for j, (b1, b2) in enumerate(edges):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                continue
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def validate_cvat_project(project: CVATProject) -> List[ValidationIssue]:
    """Run lightweight structural checks on a CVAT project."""

    issues: List[ValidationIssue] = []

    label_names = {label.name for label in project.labels}
    missing = REQUIRED_LABELS - label_names
    if missing:
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Missing required labels: {sorted(missing)}",
            )
        )

    unknown = label_names - EXPECTED_LABELS
    if unknown:
        issues.append(
            ValidationIssue(
                level="warning",
                message=f"Unknown labels present: {sorted(unknown)}",
            )
        )

    # Per-image geometry sanity checks
    for image in project.images:
        issues.extend(_validate_image_annotations(image))

    return issues


def _validate_image_annotations(image: ImageAnnotations) -> Iterable[ValidationIssue]:
    # Polylines must have at least 2 points
    for polyline in image.polylines:
        if len(polyline.points) < 2:
            yield ValidationIssue(
                level="error",
                message=f"Image {image.name}: polyline '{polyline.label}' has fewer than 2 points",
            )
            continue

        length = calculate_arc_length(polyline.to_array())
        if length < MIN_POLYLINE_LENGTH_PX:
            yield ValidationIssue(
                level="warning",
                message=(
                    f"Image {image.name}: polyline '{polyline.label}' length {length:.2f}px "
                    f"below threshold {MIN_POLYLINE_LENGTH_PX}px"
                ),
            )

    # Polygons must have at least 3 points
    for polygon in image.polygons:
        if len(polygon.points) < 3:
            yield ValidationIssue(
                level="error",
                message=f"Image {image.name}: polygon '{polygon.label}' has fewer than 3 points",
            )
            continue

        arr = polygon.to_array()
        area = calculate_polygon_area(arr)
        if area < MIN_POLYGON_AREA_PX2:
            yield ValidationIssue(
                level="warning",
                message=(
                    f"Image {image.name}: polygon '{polygon.label}' area {area:.2f}px² "
                    f"below threshold {MIN_POLYGON_AREA_PX2}px²"
                ),
            )

        if _is_self_intersecting(arr):
            yield ValidationIssue(
                level="warning",
                message=f"Image {image.name}: polygon '{polygon.label}' appears self-intersecting",
            )
