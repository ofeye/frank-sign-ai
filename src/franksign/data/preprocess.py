"""Lightweight preprocessing helpers for Frank Sign images.

These helpers avoid heavy dependencies while providing a clear extension point
for future augmentation/normalization work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("Pillow is required for preprocessing.") from exc


def preprocess_images(
    input_dir: str | Path,
    output_dir: str | Path,
    image_size: Sequence[int] = (256, 256),
    overwrite: bool = False,
) -> Tuple[int, int]:
    """Resize/copy images from ``input_dir`` to ``output_dir``.

    Args:
        input_dir: Directory containing raw images.
        output_dir: Destination for processed images.
        image_size: Target (height, width) for resizing.
        overwrite: Overwrite existing files when True.

    Returns:
        Tuple of (processed_count, skipped_count).
    """

    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for img_path in _iter_images(src):
        rel_path = img_path.relative_to(src)
        out_path = dst / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")
        image = image.resize((image_size[1], image_size[0]))
        image.save(out_path)
        processed += 1

    return processed, skipped


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in root.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            yield path
