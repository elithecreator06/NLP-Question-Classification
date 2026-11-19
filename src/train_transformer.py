"""Fine-tune DistilBERT and save an auditable evaluation report."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import PROJECT_ROOT, dataset_summary, load_processed_data, require_ground_truth


MODEL_SAVE_PATH = PROJECT_ROOT / "Models" / "distilbert-question-classifier"
REPORT_PATH = PROJECT_ROOT / "Models" / "transformer_evaluation.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "Models" / "transformer_confusion_matrix.csv"
RANDOM_SEED = 42


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(prediction):
    logits, labels = prediction
    predicted_labels = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "f1_weighted": f1_score(labels, predicted_labels, average="weighted", zero_division=0),
    }


def train_transformer(*, allow_heuristic_labels: bool = False) -> dict:
    """Fine-tune DistilBERT using a deterministic split and save evaluation evidence."""
    set_seed(RANDOM_SEED)
    df = load_processed_data()
    require_ground_truth(df, allow_heuristic_labels)

    duplicates = int(df["cleaned"].duplicated().sum())
    if duplicates:
        print(f"Removing {duplicates} duplicate cleaned question(s) before splitting.")
        df = df.drop_duplicates(subset="cleaned").copy()
    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Deduplication left fewer than two examples for a label; add more labelled data."
        )

    label_names = sorted(df["label"].unique().tolist())
    label2id = {label: index for index, label in enumerate(label_names)}
    id2label = {index: label for label, index in label2id.items()}
    labels = df["label"].map(label2id)
    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned"].tolist(),
        labels.tolist(),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = QuestionDataset(
        tokenizer(x_train, truncation=True, padding=True), y_train
    )
    test_dataset = QuestionDataset(
        tokenizer(x_test, truncation=True, padding=True), y_test
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    output = trainer.predict(test_dataset)
    predicted_ids = np.argmax(output.predictions, axis=-1)
    matrix = confusion_matrix(y_test, predicted_ids, labels=list(id2label))
    report = classification_report(
        y_test,
        predicted_ids,
        labels=list(id2label),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    pd.DataFrame(matrix, index=label_names, columns=label_names).rename_axis("actual").to_csv(
        CONFUSION_MATRIX_PATH
    )

    summary = dataset_summary(df)
    result = {
        "model": "distilbert-base-uncased",
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
    print(f"Saved evaluation report to: {REPORT_PATH}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the DistilBERT classifier.")
    parser.add_argument(
        "--allow-heuristic-labels",
        action="store_true",
        help="allow demonstration-only training on non-manual labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_transformer(allow_heuristic_labels=args.allow_heuristic_labels)
