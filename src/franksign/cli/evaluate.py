"""Placeholder evaluation CLI.

This will host post-training evaluation (metrics, XAI exports) once models are
implemented. For now it advertises the expected interface and exits cleanly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Frank Sign models (placeholder)")
    parser.add_argument(
        "--checkpoint",
        "-m",
        type=str,
        required=False,
        help="Path to model checkpoint (to be used when implemented).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to evaluation configuration (YAML).",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (future use).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    print("⚠️  Evaluation pipeline is not implemented yet.")
    print("👉 Expected next steps: load trained checkpoints, run metrics, export XAI.")
    if checkpoint:
        print(f"🧠 Provided checkpoint: {checkpoint}")
    print(f"📄 Config provided: {Path(args.config)}")
    print(f"📊 Target split: {args.split}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
