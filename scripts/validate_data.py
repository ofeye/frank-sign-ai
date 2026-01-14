#!/usr/bin/env python
"""Validate clinical CSV against a schema.

Uses ClinicalDataLoader to parse/clean the CSV, then applies a Pandera schema.
This is intentionally minimal and can be extended as production data evolves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pandera as pa
    from pandera.typing import DataFrame
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pandera is required for validation. Install dependencies from pyproject.toml."
    ) from exc

# Add src to path for development usage
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from franksign.data.clinical_loader import ClinicalDataLoader  # noqa: E402
from franksign.data.validation import ClinicalSchema, validate_cvat_project  # noqa: E402
from franksign.data.cvat_parser import load_annotations  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate clinical CSV and annotations")
    parser.add_argument(
        "--clinical",
        "-c",
        type=str,
        default="FS - AI - Sayfa1.csv",
        help="Path to clinical CSV (default: sample data).",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default=None,
        help="Optional path to CVAT annotations.xml for structural checks.",
    )
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        default=None,
        help="Optional path to save validation summary as CSV.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    csv_path = Path(args.clinical)

    try:
        loader = ClinicalDataLoader(csv_path)
        df = loader.load()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"📂 Loaded clinical data: {csv_path}")
    print(f"🧮 Rows: {len(df)} | Columns: {len(df.columns)}")

    try:
        validated: DataFrame[ClinicalSchema] = ClinicalSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        print("❌ Clinical validation failed. Top issues:")
        print(exc.failure_cases.head())
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            exc.failure_cases.to_csv(args.summary, index=False)
            print(f"📝 Failure cases saved to {args.summary}")
        return 2

    print("✅ Clinical validation passed against ClinicalSchema")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        validated.to_csv(args.summary, index=False)
        print(f"📝 Cleaned data saved to {args.summary}")

    # Optional CVAT validation
    if args.annotations:
        ann_path = Path(args.annotations)
        if not ann_path.exists():
            print(f"❌ Annotations file not found: {ann_path}")
            return 3
        project = load_annotations(ann_path)
        issues = validate_cvat_project(project)
        if issues:
            print("⚠️  CVAT validation reported:")
            for issue in issues:
                print(f" - [{issue.level}] {issue.message}")
        else:
            print("✅ CVAT validation passed basic structural checks")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
