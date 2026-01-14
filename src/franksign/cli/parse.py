"""CLI for parsing CVAT annotations and extracting geometric features.

This is a thin, maintained wrapper around the parsing utilities in
``franksign.data``. It intentionally keeps dependencies minimal and mirrors the
user-facing workflow described in README.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from franksign.data.cvat_parser import load_annotations
from franksign.data.geometric_features import (
    GeometricFeatureExtractor,
    extract_features_batch,
    features_to_dataframe,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse CVAT annotations and extract Frank Sign features",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to CVAT annotations.xml file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV path (optional). If omitted, only summary is printed.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=None,
        help="Pixels per mm (for dimensional calibration).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed label information.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"❌ Annotation file not found: {xml_path}")
        return 1

    print(f"📂 Loading annotations from: {xml_path}")
    project = load_annotations(xml_path)
    print(f"✅ Loaded {project.num_images} images")
    print(f"📊 Project: {project.name}")
    print(f"🏷️  Labels defined: {len(project.labels)}")

    if args.verbose:
        print("\nLabel types:")
        for label in project.labels:
            print(f"   - {label.name} ({label.type})")

    print("\n🔍 Extracting geometric features...")
    extractor = GeometricFeatureExtractor(scale_factor=args.scale)
    features = extract_features_batch(project.images, scale_factor=args.scale)

    num_with_fs = sum(1 for f in features if f.has_frank_sign)
    print(f"📈 Frank Sign present: {num_with_fs}/{len(features)} images")

    fs_features = [f for f in features if f.frank_sign_line is not None]
    if fs_features:
        avg_length = sum(f.frank_sign_line.length for f in fs_features) / len(fs_features)
        avg_tortuosity = sum(
            f.frank_sign_line.tortuosity for f in fs_features
        ) / len(fs_features)
        avg_curvature = sum(
            f.frank_sign_line.curvature_mean for f in fs_features
        ) / len(fs_features)
        print("\n📐 Average Frank Sign metrics (px unless scaled):")
        print(f"   - Length: {avg_length:.2f}")
        print(f"   - Tortuosity: {avg_tortuosity:.3f}")
        print(f"   - Mean curvature: {avg_curvature:.4f}")

    if args.output:
        df = features_to_dataframe(features)
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Features saved to: {output_path}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Rows: {len(df)}")

    print("\n✨ Done!")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
