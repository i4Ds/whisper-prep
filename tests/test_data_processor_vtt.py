"""
Unit tests for VTT subtitle reading and text normalization behaviors.
"""

import tempfile
import unittest
from pathlib import Path

from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.text_normalizer import normalize_text

class TestDataProcessorVTT(unittest.TestCase):
    def test_read_utterances_from_vtt_and_normalization(self):
        # Create a temporary VTT file with two cues,
        # one containing a <font> tag with starred text, the other with parentheses.
        vtt_content = (
            "WEBVTT\n\n"
            "00:00:26.710 --> 00:00:27.730\n"
            "<font color=\"#ffffff\">* Jauchzer *</font>\n\n"
            "00:00:27.730 --> 00:00:29.000\n"
            "Hello (World)\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8")
        tmp.write(vtt_content)
        tmp.flush()
        tmp.close()

        # Read utterances without unicode normalization
        utterances = DataProcessor.read_utterances_from_vtt(tmp.name)
        # Expect two utterances parsed
        self.assertEqual(len(utterances), 2)

        # Normalize their texts: first should drop <font> content, second should drop parentheses content
        normalized_texts = [normalize_text(u.text) for u in utterances]
        # First cue becomes empty (filtered out later if desired)
        self.assertEqual(normalized_texts[0], "")
        # Second cue drops '(World)' leaving only 'Hello'
        self.assertEqual(normalized_texts[1], "Hello")

        # Clean up temporary file
        Path(tmp.name).unlink()