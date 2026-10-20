import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pathlib import Path
import joblib

"""
Baseline training script for the question classification project. 

This module trains a traditional machine-learning model using TF-IDF features and Logistic Regression.
The aim is to establish a benchmark level of performance before fine-tuning a transformer-based model such as DistilBERT.

The workflow is as follows:
1. Load the processed dataset (cleaned text + preliminary labels).
2. Split the data into training and testing sets using stratification to preserve label distribution.
3. Convert text into numerical features using TF-IDF, capturing word importance across the dataset.
4. Train a Logistic Regression classifier as te baseline model.
5. Evaluate the model using standard metrics (precision, recall, F1-score).
6. Save both the trained model and the TF-IDF vectorizer for later use in predication scripts or comparison with transformer-based results. 
"""

# Keep data and artifacts inside this repository regardless of where it is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "processed_questions.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "baseline_logreg.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "Models" / "tfidf_vectorizer.pkl"

def load_processed_data():
    """Load the prepared data and check it is suitable for a fair split."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}. Run preprocess.py first."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"cleaned", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Processed dataset is missing required column(s): {missing}")

    # Drop incomplete rows before splitting so the train and test sets stay valid.
    df = df.dropna(subset=["cleaned", "label"])
    label_counts = df["label"].value_counts()
    if len(label_counts) < 2:
        raise ValueError("At least two labels are needed to train a classifier.")
    if label_counts.min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")
    return df

def train_baseline():
    print("Loading processed dataset...")
    df = load_processed_data()

    x = df["cleaned"]
    y = df["label"]

    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42, stratify = y
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features = 5000)
    x_train_vector = vectorizer.fit_transform(x_train)
    x_test_vector = vectorizer.transform(x_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter = 200)
    model.fit(x_train_vector, y_train)

    print("Evaluating model...")
    y_pred = model.predict(x_test_vector)
    # The smallest class may have no predictions, so report zero rather than warning.
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Saving model and vectorizer...")
    MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model,MODEL_PATH)
    joblib.dump(vectorizer,VECTORIZER_PATH)

    print("Done! Baseline model saved.")

if __name__ == "__main__":
    train_baseline()
