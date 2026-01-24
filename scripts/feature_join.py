#!/usr/bin/env python
"""Extract geometric features from CVAT annotations and join with clinical data.

Outputs Parquet files:
- features.parquet: per-image geometric features
- clinical_clean.parquet: cleaned clinical data
- master_features.parquet: joined features + clinical
- match_report.parquet: match/unmatched summary rows
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path for development usage
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.clinical_loader import ClinicalDataLoader, extract_patient_id_from_image  # noqa: E402
from franksign.data.cvat_parser import load_annotations  # noqa: E402
from franksign.data.geometric_features import (  # noqa: E402
    GeometricFeatureExtractor,
    extract_features_batch,
    features_to_dataframe,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join CVAT-derived features with clinical data")
    parser.add_argument("--annotations", "-a", required=True, type=str, help="Path to annotations.xml")
    parser.add_argument("--clinical", "-c", default="FS - AI - Sayfa1.csv", type=str, help="Path to clinical CSV")
    parser.add_argument("--output-dir", "-o", default="data/processed", type=str, help="Directory to write outputs")
    parser.add_argument("--scale", "-s", default=None, type=float, help="Pixels-per-mm scale (optional)")
    parser.add_argument("--report", "-r", default=None, type=str, help="Optional path to save match report (CSV/Parquet)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading annotations from {args.annotations}")
    project = load_annotations(args.annotations)
    extractor = GeometricFeatureExtractor(scale_factor=args.scale)
    features = extract_features_batch(project.images, scale_factor=args.scale)
    df_feat = features_to_dataframe(features)

    # Derive patient_id from image_name
    df_feat["patient_id"] = df_feat["image_name"].apply(extract_patient_id_from_image)

    feat_path = output_dir / "features.parquet"
    df_feat.to_parquet(feat_path, index=False)
    print(f"💾 Saved features: {feat_path} ({len(df_feat)} rows)")

    print(f"📂 Loading clinical data from {args.clinical}")
    clinical_loader = ClinicalDataLoader(args.clinical)
    df_clin = clinical_loader.load()
    df_clin["patient_id"] = df_clin["patient_id"].astype(str)

    clinical_path = output_dir / "clinical_clean.parquet"
    df_clin.to_parquet(clinical_path, index=False)
    print(f"💾 Saved cleaned clinical data: {clinical_path} ({len(df_clin)} rows)")

    # Join
    df_master = df_feat.merge(df_clin, on="patient_id", how="left", suffixes=("", "_clin"))
    master_path = output_dir / "master_features.parquet"
    df_master.to_parquet(master_path, index=False)
    print(f"💾 Saved joined master features: {master_path} ({len(df_master)} rows)")

    # Match report
    unmatched_images = df_master[df_master.isna().any(axis=1)]["image_name"].unique().tolist()
    unmatched_patients = df_clin[~df_clin["patient_id"].isin(df_master["patient_id"])]
    match_rate = 1 - (len(unmatched_images) / max(len(df_master), 1))

    report_rows = []
    report_rows.append({"metric": "total_images", "value": len(df_master)})
    report_rows.append({"metric": "total_patients", "value": len(df_clin)})
    report_rows.append({"metric": "unmatched_images", "value": len(unmatched_images)})
    report_rows.append({"metric": "unmatched_patients", "value": len(unmatched_patients)})
    report_rows.append({"metric": "match_rate", "value": match_rate})

    df_report = pd.DataFrame(report_rows)
    report_path = Path(args.report) if args.report else output_dir / "match_report.parquet"
    if report_path.suffix.lower() in {".csv"}:
        df_report.to_csv(report_path, index=False)
    else:
        df_report.to_parquet(report_path, index=False)
    print(f"📝 Match report saved to: {report_path}")

    if unmatched_images:
        print(f"⚠️ Unmatched images ({len(unmatched_images)}): {unmatched_images[:5]}{'...' if len(unmatched_images) > 5 else ''}")
    if len(unmatched_patients) > 0:
        print(f"⚠️ Patients without images: {len(unmatched_patients)}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
