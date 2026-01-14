# Data Schema Documentation

> Reference for CVAT annotation labels and their attributes

## Label Overview

| Label | Type | Color | Purpose |
|-------|------|-------|---------|
| `ear_outer_contour` | polygon | #FF0000 | Full ear boundary for normalization |
| `franks_sign_region` | polygon | #FFAA00 | Region containing Frank Sign |
| `franks_sign_line` | polyline | #FF0000 | The diagonal crease itself (primary target) |
| `ear_canal_center` | point | #FFFF00 | Reference point for alignment |
| `tragus_point` | point | #FF6600 | Anatomical landmark |
| `antitragus_point` | point | #33FF00 | Anatomical landmark |
| `antitragus_end_point` | point | #33CCFF | End of antitragus structure |
| `intertragic_notch` | point | #9933FF | Notch between tragus/antitragus |
| `earlobe_tip` | point | #0066FF | Lowest point of earlobe |
| `ear_top` | point | #FF33CC | Highest point of ear (Superaurale) |
| `earlobe_attachment_point` | point | #FF3300 | Where earlobe meets face |
| `image_quality_assessment` | point | #808080 | Image quality metadata carrier |
| `annotation_metadata` | point | #404040 | Annotator info carrier |
| `patient_metadata` | point | #008080 | Patient identification |

---

## Detailed Label Attributes

### `franks_sign_line` (Primary Target)

The most important label for segmentation and feature extraction.

| Attribute | Type | Values | Description |
|-----------|------|--------|-------------|
| `presence` | select | present, absent, questionable | Whether Frank Sign exists |
| `visibility_quality` | select | excellent, good, fair, poor | How clearly visible |
| `line_continuity` | select | continuous, interrupted, fragmented | Gap pattern |
| `depth_impression` | select | deep, moderate, shallow, barely_visible | Subjective depth |
| `line_character` | select | single_line, multiple_parallel, branched | Structure type |
| `overall_shape` | select | straight, curved, angular, s_shaped, complex | Shape category |
| `curvature_direction` | select | posterior_convex, anterior_convex, mixed, straight | Curve direction |
| `length_impression` | select | very_long, long, moderate, short, very_short | Subjective length |
| `annotator_confidence` | select | very_confident, confident, uncertain, very_uncertain | Annotator certainty |

### `ear_outer_contour`

| Attribute | Type | Values |
|-----------|------|--------|
| `ear_side` | select | right, left |
| `contour_quality` | select | excellent, good, fair, poor |
| `boundary_clarity` | select | clear, partially_obscured, difficult_to_define |

### `franks_sign_region`

| Attribute | Type | Values |
|-----------|------|--------|
| `region_visibility` | select | clear, partially_visible, obscured |
| `region_quality` | select | excellent, good, fair, poor |

### `earlobe_tip`

| Attribute | Type | Values |
|-----------|------|--------|
| `earlobe_type` | select | free_hanging, attached, semi_attached |
| `visibility` | select | clear, partially_visible, obscured |

### `image_quality_assessment`

| Attribute | Type | Values |
|-----------|------|--------|
| `overall_image_quality` | select | excellent, good, acceptable, poor, unusable |
| `lighting_quality` | select | optimal, good, suboptimal, poor, very_poor |
| `focus_sharpness` | select | very_sharp, sharp, acceptable, slightly_blurry, very_blurry |
| `ear_orientation` | select | optimal, good, acceptable, suboptimal, poor |
| `coverage_completeness` | select | complete, mostly_complete, partial, insufficient |
| `artifacts_present` | checkbox | hair_obstruction, jewelry, shadow, reflection, motion_blur, compression_artifacts |

---

## Geometric Features to Extract

From the annotation primitives, we calculate:

| Feature | Source Label | Calculation |
|---------|--------------|-------------|
| `length_mm` | `franks_sign_line` | Arc length with ruler calibration |
| `curvature_mean` | `franks_sign_line` | Mean discrete curvature |
| `curvature_max` | `franks_sign_line` | Maximum local curvature |
| `tortuosity` | `franks_sign_line` | Arc length / Euclidean distance |
| `depth_ratio` | `franks_sign_line` + `ear_outer_contour` | Relative depth estimate |
| `rel_length` | `franks_sign_line` + `ear_top` + `earlobe_tip` | Length / Ear height |
| `localization_x` | `franks_sign_line` + anatomical points | Normalized x position |
| `localization_y` | `franks_sign_line` + anatomical points | Normalized y position |
| `ear_area_mm2` | `ear_outer_contour` | Total ear area |
| `frank_region_area` | `franks_sign_region` | Frank Sign region area |

---

## Coordinate Systems

### Image Coordinates
- Origin: Top-left corner
- X: Increases rightward
- Y: Increases downward
- Units: Pixels

### Normalized Coordinates (for localization features)
- Origin: `ear_canal_center` or centroid of `ear_outer_contour`
- Normalized by ear height (`ear_top` to `earlobe_tip` distance)
- Range: [-1, 1] for both axes

---

## Future Extensions

Potential new labels to add if model performance is insufficient:

- [ ] `antihelix_line` - Antihelix curve for additional anatomy
- [ ] `concha_region` - Concha bowl area
- [ ] `wrinkle_lines` - Other wrinkles for differentiation
- [ ] `scale_ruler` - Explicit ruler marking for calibration
