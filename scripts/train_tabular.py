#!/usr/bin/env python
"""Train baseline tabular models on joined geometric + clinical features.

Default target: `syntax_score` (regression). Classification mode bins the target into tertiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline tabular model on master features")
    parser.add_argument("--features", "-f", required=True, type=str, help="Path to master_features.parquet")
    parser.add_argument("--target", "-t", default="syntax_score", type=str, help="Target column name")
    parser.add_argument("--mode", "-m", choices=["regression", "classification"], default="regression")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--output-dir", "-o", type=str, default="experiments/tabular", help="Where to save metrics")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser


def _load_data(path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    y = df[target]
    X = df.drop(columns=[target])
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number]).copy()
    # Drop rows with missing target or all-null features
    mask = y.notna() & (~X.isna().all(axis=1))
    X = X[mask]
    y = y[mask]
    # Simple median impute for remaining NaNs
    X = X.fillna(X.median())
    return X, y


def _bin_target(y: pd.Series) -> pd.Series:
    # Tertile binning; if not enough unique values, fallback to median split
    try:
        binned, bins = pd.qcut(y, q=3, labels=["low", "mid", "high"], retbins=True, duplicates="drop")
    except ValueError:
        binned, bins = pd.qcut(y, q=2, labels=["low", "high"], retbins=True, duplicates="drop")
    return binned.astype(str)


def _train_regression(X: pd.DataFrame, y: pd.Series, test_size: float, rs: int):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
    model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=rs, n_estimators=200))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": mse ** 0.5,
        "r2": float(r2_score(y_test, preds)),
    }
    return model, metrics


def _train_classification(X: pd.DataFrame, y: pd.Series, test_size: float, rs: int):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=rs)
    model = make_pipeline(StandardScaler(with_mean=False), RandomForestClassifier(random_state=rs, n_estimators=300))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }
    return model, metrics


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = _load_data(Path(args.features), args.target)

    if args.mode == "classification":
        y_proc = _bin_target(y)
        model, metrics = _train_classification(X, y_proc, test_size=args.test_size, rs=args.random_state)
    else:
        model, metrics = _train_regression(X, y, test_size=args.test_size, rs=args.random_state)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Training complete. Metrics saved to {metrics_path}")
    print(metrics)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
