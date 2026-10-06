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

DATA_PATH = Path("../Data/Processed/processed_questions.csv")
MODEL_PATH = Path("../Models/baseline_logreg.pkl")
VECTORIZER_PATH = Path("../Models/tfidf_vectorizer.pkl")

def train_baseline():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

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
    print(classification_report(y_test, y_pred))

    print("Saving model and vectorizer...")
    MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model,MODEL_PATH)
    joblib.dump(vectorizer,VECTORIZER_PATH)

    print("Done! Baseline model saved.")

if __name__ == "__main__":
    train_baseline()