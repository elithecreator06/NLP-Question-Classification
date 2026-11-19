"""Backward-compatible manual DistilBERT smoke test.

Use ``predict.py --model transformer`` for normal single-question inference.
"""

from predict import predict_transformer


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
