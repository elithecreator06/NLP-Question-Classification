# NLP Question Classification

This project compares a TF-IDF and Logistic Regression baseline with a
fine-tuned DistilBERT classifier for short questions.

## Running the project

From the repository root, install the dependencies and run the pipeline:

```powershell
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train_baseline.py
python src/train_transformer.py
python src/test_transformer.py
```

## Data note

The supplied raw `Label` column contains `unknown` for every question. During
preprocessing, the project creates temporary labels using simple keyword rules:
`conceptual`, `computational`, `procedural`, and `other`.

This is useful for demonstrating the pipeline, but the reported model metrics
show agreement with these preliminary labels rather than independently checked
ground truth. A future version should use manually labelled questions for a
fairer measure of classification quality.
