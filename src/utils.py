"""Shared paths and dataset checks for the training scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "processed_questions.csv"


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load a processed dataset and reject malformed training input early."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run preprocess.py first."
        )

    df = pd.read_csv(path)
    required_columns = {"cleaned", "label", "label_source"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Processed dataset is missing required column(s): {missing}")

    df = df.dropna(subset=["cleaned", "label", "label_source"]).copy()
    df["cleaned"] = df["cleaned"].astype(str).str.strip()
    df = df[df["cleaned"].ne("")]
    if df.empty:
        raise ValueError("The processed dataset does not contain any usable questions.")

    label_counts = df["label"].value_counts()
    if len(label_counts) < 2:
        raise ValueError("At least two labels are needed to train a classifier.")
    if label_counts.min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")
    return df


def require_ground_truth(df: pd.DataFrame, allow_heuristic_labels: bool) -> None:
    """Prevent placeholder labels from being mistaken for a quality result."""
    sources = set(df["label_source"].astype(str).str.lower())
    if sources == {"manual"}:
        return
    if allow_heuristic_labels:
        print(
            "WARNING: training with heuristic labels. Metrics measure agreement with "
            "the labeling rules, not independently verified classification quality."
        )
        return
    raise ValueError(
        "This dataset does not contain only manual labels. Label the raw data first, "
        "or pass --allow-heuristic-labels for a demonstration-only run."
    )


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return serialisable provenance for a training/evaluation report."""
    return {
        "examples": len(df),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "label_sources": sorted(set(df["label_source"].astype(str))),
        "ground_truth": set(df["label_source"].astype(str).str.lower()) == {"manual"},
    }
