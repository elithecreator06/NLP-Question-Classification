import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from pathlib import Path
import inspect

"""
Transformer fine-tuning script for the question classification project.

This module trains a DistilBERT model on the processed dataset to achieve state-of-the-art performance in classifying conceptual, procedural, computational, and other questions types.
DistilBERT is a lightweight transformer model that retains strong language understanding while being computationally efficient.

The workflow is as follows:
1. Load the processed dataset containing cleaned text and preliminary labels.
2. Encode labels numerically using LabelEncoder for compatibility with PyTorch.
3. Tokenise the text using DistilBERT's tokenizer, converting each question into input IDs and attention masks.
4. Wrap the encoded data into a custom PyTorch Dataset class.
5. Load a pretrained DistilBERT model and configure training parameters such as learning rate, batch size, number of epochs, and evaluation strategy.
6. Fine-tune the model using HuggingFace's Trainer API, which handles batching, optimisation, evaluation, and checkpointing.
7. Save the final fine-tuned model and tokenizer for later inference or comparison with the baseline TF-IDF model.

This script forms the core of the modern NLP component of the project and provides an improvement over traditional machine-learning methods.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "processed_questions.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "Models" / "distilbert-question-classifier"

def load_data(data_path: Path = DATA_PATH):
    """Load validated processed data and encode its labels."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run preprocess.py first."
        )
    df = pd.read_csv(data_path)
    if not {"cleaned", "label"}.issubset(df.columns):
        raise ValueError("Processed dataset must contain 'cleaned' and 'label' columns.")
    df = df.dropna(subset=["cleaned", "label"])
    if df["label"].nunique() < 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Each of at least two labels needs two examples for a stratified split.")
    x = df["cleaned"].tolist()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"]).tolist()
    label_map = dict(enumerate(label_encoder.classes_))
    return x, y, label_map

class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(prediction):
    """Calculate accuracy and weighted F1 during evaluation."""
    logits, labels = prediction
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def train_transformer(
    data_path: Path = DATA_PATH,
    model_save_path: Path = MODEL_SAVE_PATH,
    base_model_name: str = "distilbert-base-uncased",
    epochs: float = 3,
    batch_size: int = 8,
    max_length: int = 128,
):
    """Fine-tune DistilBERT and save the model, tokenizer, and label mapping."""
    if epochs <= 0 or batch_size <= 0 or max_length <= 0:
        raise ValueError("epochs, batch_size, and max_length must all be positive.")

    texts, labels, label_map = load_data(data_path)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=max_length)
    model = DistilBertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(label_map),
        id2label=label_map,
        label2id={label: index for index, label in label_map.items()},
    )

    model_save_path = Path(model_save_path)
    arguments = {
        "output_dir": str(model_save_path / "checkpoints"),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "report_to": "none",
        "seed": 42,
    }
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        arguments["eval_strategy"] = "epoch"
    else:
        arguments["evaluation_strategy"] = "epoch"
    trainer_options = {
        "model": model,
        "args": TrainingArguments(**arguments),
        "train_dataset": QuestionDataset(train_encodings, train_labels),
        "eval_dataset": QuestionDataset(eval_encodings, eval_labels),
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_options["processing_class"] = tokenizer
    else:
        trainer_options["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_options)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(model_save_path)
    return metrics


if __name__ == "__main__":
    metrics = train_transformer()
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Done! Transformer model saved to:", MODEL_SAVE_PATH)
