"""Dataset scaffolding for Frank Sign images and clinical data.

This skeleton is intentionally lightweight and future-proof:
- Uses torch Dataset when available.
- Provides hooks for transforms/augmentations without enforcing a stack yet.
- Links images to clinical records via patient ID extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torch is required for FrankSignDataset. Install dependencies from pyproject.toml."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("Pillow is required to load images.") from exc

from franksign.data.clinical_loader import extract_patient_id_from_image


@dataclass
class Sample:
    """Container for a single sample returned by the dataset."""

    image: Any
    meta: Dict[str, Any]


class FrankSignDataset(Dataset):
    """Minimal torch Dataset for Frank Sign images.

    Args:
        image_paths: Sequence of image file paths.
        transform: Optional callable applied to PIL images.
        clinical_df: Optional pandas DataFrame with clinical data; if provided,
            samples include a matched clinical record when patient IDs align.
    """

    def __init__(
        self,
        image_paths: Sequence[Path | str],
        transform: Optional[Callable[[Any], Any]] = None,
        clinical_df: Any | None = None,
    ) -> None:
        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        self.transform = transform
        self.clinical_df = clinical_df

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        meta: Dict[str, Any] = {"image_path": str(image_path)}

        patient_id = extract_patient_id_from_image(image_path.name)
        if patient_id:
            meta["patient_id"] = patient_id
            if self.clinical_df is not None and "patient_id" in self.clinical_df.columns:
                matches = self.clinical_df[
                    self.clinical_df["patient_id"].astype(str) == str(patient_id)
                ]
                if len(matches) > 0:
                    meta["clinical_record"] = matches.iloc[0].to_dict()

        return Sample(image=image, meta=meta)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        split_files: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        clinical_df: Any | None = None,
    ) -> "FrankSignDataset":
        """Create dataset from a config dict (mirrors configs/default.yaml).

        Args:
            config: Parsed YAML configuration.
            split_files: Optional list of image file names to restrict to a split.
            transform: Optional transform callable.
            clinical_df: Optional clinical dataframe for joining records.
        """

        images_dir = Path(config["data"]["images_dir"])
        if split_files is None:
            image_paths = sorted(images_dir.glob("**/*"))
        else:
            image_paths = [images_dir / name for name in split_files]

        image_paths = [p for p in image_paths if p.is_file()]
        if not image_paths:
            raise FileNotFoundError(
                f"No images found under {images_dir}. Place sample data or adjust config."
            )

        return cls(image_paths=image_paths, transform=transform, clinical_df=clinical_df)
