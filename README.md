# NLP Question Classification

This is a college NLP project that classifies short questions by their type.
It compares two models: a TF-IDF + Logistic Regression baseline and a
fine-tuned DistilBERT model.

The possible question types are:

- `conceptual` — asks for an explanation, definition, comparison, or purpose
- `computational` — asks for a calculation or a specific value
- `procedural` — asks how to do something step by step
- `other` — does not fit the other categories

## Getting started

From the project folder, create a virtual environment, activate it, and install
the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the models

Run these commands to clean the data and train both models:

```powershell
python src/preprocess.py
python src/train_baseline.py
python src/train_transformer.py
```

The trained models, evaluation reports, and confusion matrices are saved in
the `Models/` folder. The DistilBERT model may take longer to train than the
baseline model.

## Using the sample labels

If the dataset only has placeholder labels, you can still run a demonstration
with the commands below:

```powershell
python src/preprocess.py --allow-heuristic-labels
python src/train_baseline.py --allow-heuristic-labels
python src/train_transformer.py --allow-heuristic-labels
```

These labels are created with simple keyword rules, so the results are only for
demonstrating that the pipeline works. They should not be treated as a final
measure of model accuracy.

## Make a prediction

```powershell
python src/predict.py "What is a binary search tree?"
python src/predict.py --model transformer "Describe the steps of breadth-first search."
```

The first command uses the baseline model. Add `--model transformer` to use
DistilBERT instead. Each command prints the predicted category and confidence.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

The tests check the text-cleaning and labeling parts of the project.
