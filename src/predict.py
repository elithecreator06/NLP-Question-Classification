"""Classify one question with a saved baseline or transformer model."""

from __future__ import annotations

import argparse
import json

import joblib
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from utils import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "Models"


def predict_baseline(question: str) -> dict:
    model_path = MODELS_DIR / "baseline_logreg.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not model_path.is_file() or not vectorizer_path.is_file():
        raise FileNotFoundError("Baseline artifacts are missing. Run train_baseline.py first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    probabilities = model.predict_proba(vectorizer.transform([question]))[0]
    index = int(probabilities.argmax())
    return {"label": str(model.classes_[index]), "confidence": float(probabilities[index])}


def predict_transformer(question: str) -> dict:
    model_path = MODELS_DIR / "distilbert-question-classifier"
    if not model_path.is_dir():
        raise FileNotFoundError("Transformer artifacts are missing. Run train_transformer.py first.")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=0)
    index = int(torch.argmax(probabilities).item())
    return {
        "label": str(model.config.id2label[index]),
        "confidence": float(probabilities[index].item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a question with a saved model.")
    parser.add_argument("question", help="question to classify")
    parser.add_argument("--model", choices=("baseline", "transformer"), default="baseline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = args.question.strip()
    if not question:
        raise ValueError("Question cannot be blank.")
    result = (
        predict_baseline(question)
        if args.model == "baseline"
        else predict_transformer(question)
    )
    print(json.dumps({"model": args.model, "question": question, **result}, indent=2))
