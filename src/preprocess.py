from pathlib import Path
import re

import pandas as pd

"""
Preprocessing script for the question classification project. 

This module prepares the raw dataset for both the baseline model and the DislBert fine-tuning stage. It performs two main tasks:
1. Text Cleaning (clean_text):
    - Convert question to lowercase
    - Removes unnecessary characters and collapses whiteshape
    - Produces a consistent text format for downstream models
    
2. Rule-Based Labelling (assign_label):
    - Applies a simple heuristic to assign each question a preliminary label
    - Labels include: conceptual, computational, procedural, and other
    - These labels act as placeholders until manual annotation or model-based refinement is applied later in the pipeline
    
The script loads the raw questions.csv file, processes each entry, and saves the cleaned and labelled dataset to processed_questions.csv. 
This ensures that both TF-IDF baseline model and the DistilBERT classifier train on a standardised and reproducible dataset.
"""

# Resolve paths from this file so the scripts work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "Raw" / "questions.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "processed_questions.csv"

def clean_text(text: str) -> str:
    """Basic text cleaning for question classification."""
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove weird chars
    text = re.sub(r"[^a-z0-9 ?!.,]", "", text)
    return text.strip()

def assign_label(question: str) -> str:
    """
    Placeholder rule-based label assignment.
    Will replace this with manual annotation or a better heuristic.
    """
    if any(word in question.lower() for word in ["define", "explain", "what is"]):
        return "conceptual"
    if any(word in question.lower() for word in ["calculate", "compute", "find"]):
        return "computational"
    if any(word in question.lower() for word in ["steps", "procedure", "how do"]):
        return "procedural"
    # Keep unmatched questions rather than silently dropping useful training data.
    return "other"

def preprocess(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Create a cleaned, preliminary-labelled dataset and return it."""
    print("Loading raw data...")
    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # The supplied CSV uses ``Question``; accepting case variants avoids a
    # brittle dependency on presentation-only column casing.
    columns = {str(column).casefold(): column for column in df.columns}
    if "question" not in columns:
        raise ValueError("Raw dataset must contain a 'Question' column.")
    question_column = columns["question"]
    if df[question_column].dropna().empty:
        raise ValueError("Raw dataset has no non-empty questions.")

    print("Cleaning data...")
    # Standardise the name used by all downstream pipeline stages.
    df = df.rename(columns={question_column: "question"})
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].ne("")].copy()
    df["cleaned"] = df["question"].apply(clean_text)

    print("Assigning labels...")
    df["label"] = df["cleaned"].apply(assign_label)

    print("Saving processed dataset...")
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)

    print("Done! Processed dataset saved to: ", processed_data_path)
    return df

if __name__ == "__main__":
    preprocess()
