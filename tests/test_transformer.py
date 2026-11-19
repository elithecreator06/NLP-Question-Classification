"""Manual DistilBERT smoke test.

Run this file directly after training a transformer model. For normal
single-question inference, use ``src/predict.py --model transformer`` instead.
"""

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from predict import predict_transformer  # noqa: E402


if __name__ == "__main__":
    test_questions = [
        "What is a binary search tree?",
        "Explain the purpose of recursion.",
        "Calculate the time complexity of merge sort.",
        "Describe the steps of BFS.",
    ]
    for question in test_questions:
        result = predict_transformer(question)
        print(f"Q: {question}\nPredicted label: {result['label']}\n")
