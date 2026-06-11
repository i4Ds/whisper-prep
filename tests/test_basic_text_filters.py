import tempfile
import unittest
from pathlib import Path

import pandas as pd

from whisper_prep import _apply_basic_text_filters


class TestBasicTextFilters(unittest.TestCase):
    def test_min_text_words_and_empty_text_filters_are_configurable(self):
        df = pd.DataFrame(
            {
                "text": [
                    "",
                    "tiny sample",
                    "one two three four five six seven eight nine",
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_folder = Path(tmpdir)
            filtered = _apply_basic_text_filters(
                df.copy(),
                out_folder,
                min_text_words=8,
                drop_empty_text=True,
            )
            self.assertEqual(
                filtered["text"].tolist(),
                ["one two three four five six seven eight nine"],
            )

            retained = _apply_basic_text_filters(
                df.copy(),
                out_folder,
                min_text_words=None,
                drop_empty_text=False,
            )
            self.assertEqual(retained["text"].tolist(), df["text"].tolist())
