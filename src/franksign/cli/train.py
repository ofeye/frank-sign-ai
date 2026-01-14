"""Placeholder training CLI.

This placeholder documents the intended interface for future training loops.
It exits with code 0 after printing guidance, so CI and packaging stay healthy
until the full training pipeline is implemented.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Frank Sign models (placeholder)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to training configuration (YAML).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="experiments",
        help="Output directory for experiments/logs (future use).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    print("⚠️  Training pipeline is not implemented yet.")
    print(
        "👉 Expected next steps: add Dataset/DataLoader, model factory, trainer, and logging."
    )
    print(f"📄 Config provided: {config_path}")
    print(f"📂 Intended output dir: {Path(args.output)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
