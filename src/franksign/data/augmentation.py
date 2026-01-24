"""Image augmentation pipeline for Frank Sign segmentation training.

This module provides config-driven augmentation using Albumentations.
Supports both image-only and image+mask augmentation for segmentation tasks.

Example:
    >>> from franksign.data.augmentation import create_augmentation_pipeline
    >>> transform = create_augmentation_pipeline(config["data"]["augmentation"])
    >>> augmented = transform(image=image, mask=mask)
    >>> aug_image, aug_mask = augmented["image"], augmented["mask"]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ============================================================
# AUGMENTATION PIPELINE BUILDER
# ============================================================

def create_augmentation_pipeline(
    config: Dict[str, Any],
    mode: str = "train",
    include_normalize: bool = True,
    include_to_tensor: bool = False,
) -> "A.Compose":
    """Create augmentation pipeline from config.
    
    Args:
        config: Augmentation config dict with keys like:
            - enabled: bool
            - horizontal_flip: bool
            - rotation_range: int (degrees)
            - scale_range: [min, max]
            - brightness_range: [min, max]
            - contrast_range: [min, max]
            - elastic_transform: bool
        mode: "train" applies augmentation, "val"/"test" only applies normalize
        include_normalize: Whether to include ImageNet normalization
        include_to_tensor: Whether to convert to PyTorch tensor
    
    Returns:
        Albumentations Compose pipeline
    
    Raises:
        ImportError: If albumentations is not installed
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for augmentation. "
            "Install with: pip install albumentations"
        )
    
    transforms: List[A.BasicTransform] = []
    
    # Only apply augmentation during training
    if mode == "train" and config.get("enabled", True):
        transforms.extend(_build_geometric_transforms(config))
        transforms.extend(_build_color_transforms(config))
    
    # Always apply normalization (ImageNet defaults)
    if include_normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    
    # Optionally convert to PyTorch tensor
    if include_to_tensor:
        transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def _build_geometric_transforms(config: Dict[str, Any]) -> List["A.BasicTransform"]:
    """Build geometric augmentation transforms."""
    transforms = []
    
    # Horizontal flip (disabled by default for ears - left/right matter)
    if config.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    # Rotation
    rotation_range = config.get("rotation_range", 15)
    if rotation_range > 0:
        transforms.append(
            A.Rotate(
                limit=rotation_range,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,
                mask_value=0,
                p=0.5,
            )
        )
    
    # Scale (zoom in/out)
    scale_range = config.get("scale_range", [0.9, 1.1])
    if scale_range and scale_range != [1.0, 1.0]:
        transforms.append(
            A.RandomScale(
                scale_limit=(scale_range[0] - 1, scale_range[1] - 1),
                p=0.5,
            )
        )
    
    # Elastic transform (disabled by default)
    if config.get("elastic_transform", False):
        transforms.append(
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                p=0.3,
            )
        )
    
    # Shift and scale combined
    transforms.append(
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0,
            rotate_limit=0,
            border_mode=0,
            p=0.3,
        )
    )
    
    return transforms


def _build_color_transforms(config: Dict[str, Any]) -> List["A.BasicTransform"]:
    """Build color/intensity augmentation transforms."""
    transforms = []
    
    # Brightness and contrast
    brightness_range = config.get("brightness_range", [0.8, 1.2])
    contrast_range = config.get("contrast_range", [0.8, 1.2])
    
    if brightness_range or contrast_range:
        brightness_limit = (
            brightness_range[0] - 1,
            brightness_range[1] - 1,
        ) if brightness_range else (0, 0)
        
        contrast_limit = (
            contrast_range[0] - 1,
            contrast_range[1] - 1,
        ) if contrast_range else (0, 0)
        
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            )
        )
    
    # Additional color augmentations
    transforms.extend([
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])
    
    return transforms


# ============================================================
# PRESET PIPELINES
# ============================================================

def get_train_transform(
    image_size: Tuple[int, int] = (256, 256),
    config: Optional[Dict[str, Any]] = None,
) -> "A.Compose":
    """Get training augmentation pipeline with sensible defaults.
    
    Args:
        image_size: Target (height, width)
        config: Optional augmentation config override
    
    Returns:
        Training augmentation pipeline
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required")
    
    default_config = {
        "enabled": True,
        "horizontal_flip": False,
        "rotation_range": 15,
        "scale_range": [0.9, 1.1],
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "elastic_transform": False,
    }
    
    if config:
        default_config.update(config)
    
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]
    
    # Add augmentation transforms
    transforms.extend(_build_geometric_transforms(default_config))
    transforms.extend(_build_color_transforms(default_config))
    
    # Normalize
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transform(
    image_size: Tuple[int, int] = (256, 256),
) -> "A.Compose":
    """Get validation/test transform (no augmentation).
    
    Args:
        image_size: Target (height, width)
    
    Returns:
        Validation transform pipeline
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required")
    
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ============================================================
# MASK GENERATION UTILITIES
# ============================================================

def create_mask_from_polygon(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    fill_value: int = 1,
) -> np.ndarray:
    """Create binary mask from polygon points.
    
    Args:
        points: Array of shape (N, 2) with x,y coordinates
        image_shape: (height, width) of output mask
        fill_value: Value to fill polygon with
    
    Returns:
        Binary mask of shape (height, width)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for mask generation")
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], fill_value)
    return mask


def create_mask_from_polyline(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    thickness: int = 3,
    fill_value: int = 1,
) -> np.ndarray:
    """Create mask from polyline (for Frank Sign line).
    
    Args:
        points: Array of shape (N, 2) with x,y coordinates
        image_shape: (height, width) of output mask
        thickness: Line thickness in pixels
        fill_value: Value for the line
    
    Returns:
        Mask with line drawn
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for mask generation")
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = points.astype(np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=fill_value, thickness=thickness)
    return mask


def combine_masks(
    line_mask: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
    num_classes: int = 3,
) -> np.ndarray:
    """Combine line and region masks into multi-class mask.
    
    Class mapping:
        0: Background
        1: Frank Sign line
        2: Frank Sign region
    
    Args:
        line_mask: Binary mask for Frank Sign line
        region_mask: Optional binary mask for Frank Sign region
        num_classes: Number of output classes
    
    Returns:
        Multi-class mask of same shape
    """
    combined = np.zeros_like(line_mask, dtype=np.uint8)
    
    # Region first (can be overwritten by line)
    if region_mask is not None:
        combined[region_mask > 0] = 2
    
    # Line on top
    combined[line_mask > 0] = 1
    
    return combined


if __name__ == "__main__":
    print("Augmentation module loaded.")
    print(f"Albumentations available: {HAS_ALBUMENTATIONS}")
