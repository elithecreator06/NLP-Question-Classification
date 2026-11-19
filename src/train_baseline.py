"""Train and evaluate a reproducible TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import PROJECT_ROOT, dataset_summary, load_processed_data, require_ground_truth


MODEL_PATH = PROJECT_ROOT / "Models" / "baseline_logreg.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "Models" / "tfidf_vectorizer.pkl"
REPORT_PATH = PROJECT_ROOT / "Models" / "baseline_evaluation.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "Models" / "baseline_confusion_matrix.csv"
RANDOM_SEED = 42


def train_baseline(*, allow_heuristic_labels: bool = False) -> dict:
    """Fit the baseline and write models plus an auditable evaluation report."""
    df = load_processed_data()
    require_ground_truth(df, allow_heuristic_labels)

    # Identical normalized questions across train/test inflate the apparent score.
    duplicates = int(df["cleaned"].duplicated().sum())
    if duplicates:
        print(f"Removing {duplicates} duplicate cleaned question(s) before splitting.")
        df = df.drop_duplicates(subset="cleaned").copy()
    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Deduplication left fewer than two examples for a label; add more labelled data."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
    x_train_vector = vectorizer.fit_transform(x_train)
    x_test_vector = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(x_train_vector, y_train)

    labels = sorted(model.classes_.tolist())
    predictions = model.predict(x_test_vector)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual").to_csv(
        CONFUSION_MATRIX_PATH
    )
    summary = dataset_summary(df)
    result = {
        "model": "TF-IDF (1-2 gram) + class-balanced Logistic Regression",
        "random_seed": RANDOM_SEED,
        "test_examples": len(y_test),
        "deduplicated_questions": duplicates,
        "dataset": summary,
        "classification_report": report,
        "warning": (
            "Metrics use heuristic labels and are demonstration-only."
            if not summary["ground_truth"]
            else None
        ),
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved evaluation report to: {REPORT_PATH}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline.")
    parser.add_argument(
        "--allow-heuristic-labels",
        action="store_true",
        help="allow demonstration-only training on non-manual labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(allow_heuristic_labels=args.allow_heuristic_labels)
