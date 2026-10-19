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

RAW_DATA_PATH = Path("../Data/Raw/questions.csv")
PROCESSED_DATA_PATH = Path("../Data/Processed/processed_questions.csv")

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
    return "other"

def preprocess():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning data...")
    df["cleaned"] = df["question"].apply(clean_text)

    print("Assigning labels...")
    df["label"] = df["cleaned"].apply(assign_label)

    print("Saving processed dataset...")
    PROCESSED_DATA_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(PROCESSED_DATA_PATH, index = False)

    print("Done! Processed dataset saved to: ", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    preprocess()