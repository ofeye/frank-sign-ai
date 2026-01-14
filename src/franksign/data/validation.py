"""Validation utilities for clinical data and CVAT annotations.

- Clinical validation uses Pandera to enforce basic schema checks.
- CVAT validation performs lightweight structural checks (labels and geometry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandera as pa
from pandera.typing import DataFrame, Series

from franksign.data.cvat_parser import CVATProject, ImageAnnotations


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


class ClinicalSchema(pa.DataFrameModel):
    """Minimal clinical schema (extend as production data grows)."""

    patient_id: Series[str] = pa.Field(nullable=False)
    fs_right: Series[Optional[int]] = pa.Field(nullable=True, isin=[0, 1, None])
    fs_left: Series[Optional[int]] = pa.Field(nullable=True, isin=[0, 1, None])
    gender: Series[Optional[str]] = pa.Field(nullable=True, isin=["M", "F", None])
    age: Series[Optional[int]] = pa.Field(nullable=True, ge=0, le=120)
    hdl: Series[Optional[float]] = pa.Field(nullable=True, ge=0)
    ldl: Series[Optional[float]] = pa.Field(nullable=True, ge=0)
    total_cholesterol: Series[Optional[float]] = pa.Field(nullable=True, ge=0)
    triglycerides: Series[Optional[float]] = pa.Field(nullable=True, ge=0)
    ef: Series[Optional[float]] = pa.Field(nullable=True, ge=0, le=100)
    syntax_score: Series[Optional[float]] = pa.Field(nullable=True, ge=0)

    class Config:
        coerce = True


@dataclass
class ValidationIssue:
    level: str  # e.g., "warning" or "error"
    message: str


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

    # Polygons must have at least 3 points
    for polygon in image.polygons:
        if len(polygon.points) < 3:
            yield ValidationIssue(
                level="error",
                message=f"Image {image.name}: polygon '{polygon.label}' has fewer than 3 points",
            )
