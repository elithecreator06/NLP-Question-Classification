from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from preprocess import clean_text, preprocess  # noqa: E402


class PreprocessTests(unittest.TestCase):
    def test_clean_text_normalises_whitespace_and_symbols(self):
        self.assertEqual(clean_text("  Explain  Naïve\nTrees!  "), "explain nave trees!")

    def test_preprocess_preserves_complete_manual_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            output = Path(directory) / "processed.csv"
            pd.DataFrame(
                {
                    "Question": ["What is a tree?", "Compute 2 + 2"],
                    "Label": ["Conceptual", "Computational"],
                }
            ).to_csv(raw, index=False)

            result = preprocess(raw, output)

            self.assertEqual(result["label"].tolist(), ["conceptual", "computational"])
            self.assertEqual(result["label_source"].unique().tolist(), ["manual"])
            self.assertTrue(output.is_file())

    def test_preprocess_rejects_unknown_labels_without_demo_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            pd.DataFrame({"Question": ["What is a tree?"], "Label": ["unknown"]}).to_csv(
                raw, index=False
            )

            with self.assertRaisesRegex(ValueError, "reviewed class"):
                preprocess(raw, Path(directory) / "processed.csv")

    def test_preprocess_marks_demo_labels_as_heuristic(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            pd.DataFrame({"Question": ["Explain recursion"], "Label": ["unknown"]}).to_csv(
                raw, index=False
            )

            result = preprocess(
                raw, Path(directory) / "processed.csv", allow_heuristic_labels=True
            )

            self.assertEqual(result.loc[0, "label"], "conceptual")
            self.assertEqual(result.loc[0, "label_source"], "heuristic")


if __name__ == "__main__":
    unittest.main()
