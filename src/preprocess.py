import pandas as pd
import re
from pathlib import Path

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