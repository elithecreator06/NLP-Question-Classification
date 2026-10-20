import pandas as pd
import re
from pathlib import Path

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

# Derive paths from this file rather than the terminal's current directory.
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
    # Keeping an "other" group means unfamiliar questions are not thrown away.
    return "other"

def preprocess():
    print("Loading raw data...")
    # Fail early with a useful message instead of a long Pandas traceback.
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    if "Question" not in df.columns:
        raise ValueError("The raw dataset needs a 'Question' column.")

    # Empty questions cannot teach either model anything, so leave them out.
    df = df.dropna(subset=["Question"])
    df["Question"] = df["Question"].astype(str).str.strip()
    df = df[df["Question"].ne("")].copy()
    if df.empty:
        raise ValueError("The raw dataset does not contain any usable questions.")

    print("Cleaning data...")
    df["cleaned"] = df["Question"].apply(clean_text)

    print("Assigning labels...")
    df["label"] = df["cleaned"].apply(assign_label)

    print("Saving processed dataset...")
    PROCESSED_DATA_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(PROCESSED_DATA_PATH, index = False)

    print("Done! Processed dataset saved to: ", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    preprocess()
