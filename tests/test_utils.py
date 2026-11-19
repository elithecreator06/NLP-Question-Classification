from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils import load_processed_data, require_ground_truth  # noqa: E402


class DatasetSafetyTests(unittest.TestCase):
    def test_heuristic_data_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.csv"
            pd.DataFrame(
                {
                    "cleaned": ["one", "two", "three", "four"],
                    "label": ["a", "a", "b", "b"],
                    "label_source": ["heuristic"] * 4,
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "manual labels"):
                require_ground_truth(load_processed_data(path), allow_heuristic_labels=False)

    def test_manual_data_is_training_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.csv"
            pd.DataFrame(
                {
                    "cleaned": ["one", "two", "three", "four"],
                    "label": ["a", "a", "b", "b"],
                    "label_source": ["manual"] * 4,
                }
            ).to_csv(path, index=False)

            require_ground_truth(load_processed_data(path), allow_heuristic_labels=False)


if __name__ == "__main__":
    unittest.main()
