"""
Unit tests for VTT subtitle reading and text normalization behaviors.
"""

import tempfile
import unittest
from pathlib import Path

from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.text_normalizer import normalize_text

class TestDataProcessorVTT(unittest.TestCase):
    def test_read_utterances_from_vtt_strips_multiline_font_tags(self):
        vtt_content = (
            "WEBVTT\n\n"
            "406\n"
            "00:25:07.640 --> 00:25:10.240\n"
            "<font color=\"#008000\">Mir tat nur beim Gedanken daran</font>\n"
            "<font color=\"#008000\">wieder alles weh.</font>\n\n"
            "407\n"
            "00:25:10.320 --> 00:25:13.000 align:start position:10%\n"
            "<i>Dann ging es weiter.</i>\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8")
        tmp.write(vtt_content)
        tmp.flush()
        tmp.close()

        utterances = DataProcessor.read_utterances_from_vtt(tmp.name)

        self.assertEqual(len(utterances), 2)
        self.assertEqual(
            utterances[0].text,
            "Mir tat nur beim Gedanken daran wieder alles weh.",
        )
        self.assertEqual(utterances[0].start, 1507640)
        self.assertEqual(utterances[0].end, 1510240)
        self.assertEqual(utterances[1].text, "Dann ging es weiter.")
        self.assertEqual(utterances[1].end, 1513000)

        Path(tmp.name).unlink()

    def test_read_utterances_from_vtt_strips_font_tags(self):
        vtt_content = (
            "WEBVTT\n\n"
            "00:29:05.400 --> 00:29:07.560\n"
            "<font color=\"#00ffff\">weil das wach hält.</font> <font\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8")
        tmp.write(vtt_content)
        tmp.flush()
        tmp.close()

        utterances = DataProcessor.read_utterances_from_vtt(tmp.name)

        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0].text, "weil das wach hält.")

        Path(tmp.name).unlink()

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
