"""Clean questions and prepare labels for model training.

Manual labels in the raw ``Label`` column are preserved. The keyword labels are
available only as an explicitly requested demonstration fallback.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "Raw" / "questions.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "processed_questions.csv"


def clean_text(text: str) -> str:
    """Apply conservative normalisation without removing useful question content."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ?!.,]", "", text)
    return text.strip()


def assign_heuristic_label(question: str) -> str:
    """Return a demonstration-only label when manually annotated data is unavailable."""
    question = question.lower()
    if any(phrase in question for phrase in ["define", "explain", "what is"]):
        return "conceptual"
    if any(phrase in question for phrase in ["calculate", "compute", "find"]):
        return "computational"
    if any(phrase in question for phrase in ["steps", "procedure", "how do"]):
        return "procedural"
    return "other"


def _manual_labels(raw_labels: pd.Series) -> pd.Series | None:
    labels = raw_labels.fillna("").astype(str).str.strip().str.lower()
    invalid = labels.isin({"", "unknown", "nan", "none", "n/a"})
    return None if invalid.any() else labels


def preprocess(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
    *,
    allow_heuristic_labels: bool = False,
) -> pd.DataFrame:
    """Write cleaned data, requiring manual labels unless demo mode is selected."""
    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")

    df = pd.read_csv(raw_data_path)
    if "Question" not in df.columns:
        raise ValueError("The raw dataset needs a 'Question' column.")

    df = df.dropna(subset=["Question"]).copy()
    df["Question"] = df["Question"].astype(str).str.strip()
    df = df[df["Question"].ne("")].copy()
    if df.empty:
        raise ValueError("The raw dataset does not contain any usable questions.")

    df["cleaned"] = df["Question"].map(clean_text)
    manual_labels = _manual_labels(df["Label"]) if "Label" in df.columns else None
    if manual_labels is not None:
        df["label"] = manual_labels
        df["label_source"] = "manual"
    elif allow_heuristic_labels:
        df["label"] = df["cleaned"].map(assign_heuristic_label)
        df["label_source"] = "heuristic"
    else:
        raise ValueError(
            "Every raw Label must be a reviewed class (not 'unknown'). Complete "
            "manual annotation first, or pass --allow-heuristic-labels for a demo run."
        )

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Saved {len(df)} processed questions to: {processed_data_path}")
    print(df["label"].value_counts().to_string())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and label question data.")
    parser.add_argument(
        "--allow-heuristic-labels",
        action="store_true",
        help="create demonstration-only keyword labels when reviewed labels are unavailable",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(allow_heuristic_labels=args.allow_heuristic_labels)
