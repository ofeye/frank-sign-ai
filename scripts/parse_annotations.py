#!/usr/bin/env python
"""Parse CVAT annotations and extract geometric features.

Usage:
    python scripts/parse_annotations.py --input annotations.xml
    python scripts/parse_annotations.py --input annotations.xml --output features.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.cvat_parser import CVATParser, load_annotations
from franksign.data.geometric_features import (
    GeometricFeatureExtractor,
    extract_features_batch,
    features_to_dataframe,
)


def main():
    parser = argparse.ArgumentParser(
        description="Parse CVAT annotations and extract Frank Sign features"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to CVAT annotations.xml file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path (default: print summary only)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=float,
        default=None,
        help="Pixels per mm (for dimensional calibration)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Parse annotations
    print(f"📂 Loading annotations from: {args.input}")
    
    try:
        project = load_annotations(args.input)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print(f"✅ Loaded {project.num_images} images")
    print(f"📊 Project: {project.name}")
    print(f"🏷️  Labels defined: {len(project.labels)}")
    
    if args.verbose:
        print("\nLabel types:")
        for label in project.labels:
            print(f"   - {label.name} ({label.type})")
    
    # Extract features
    print(f"\n🔍 Extracting geometric features...")
    features = extract_features_batch(project.images, scale_factor=args.scale)
    
    # Summary
    num_with_fs = sum(1 for f in features if f.has_frank_sign)
    print(f"📈 Frank Sign present: {num_with_fs}/{len(features)} images")
    
    # Calculate average features for images with Frank Sign
    fs_features = [f for f in features if f.frank_sign_line is not None]
    if fs_features:
        avg_length = sum(f.frank_sign_line.length for f in fs_features) / len(fs_features)
        avg_tortuosity = sum(f.frank_sign_line.tortuosity for f in fs_features) / len(fs_features)
        avg_curvature = sum(f.frank_sign_line.curvature_mean for f in fs_features) / len(fs_features)
        
        print(f"\n📐 Average Frank Sign metrics (N={len(fs_features)}):")
        print(f"   - Length: {avg_length:.2f} px")
        print(f"   - Tortuosity: {avg_tortuosity:.3f}")
        print(f"   - Mean curvature: {avg_curvature:.4f}")
    
    # Export to CSV
    if args.output:
        df = features_to_dataframe(features)
        df.to_csv(args.output, index=False)
        print(f"\n💾 Features saved to: {args.output}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Rows: {len(df)}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
