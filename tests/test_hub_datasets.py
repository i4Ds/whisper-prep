"""Integration tests: verify uploaded datasets are reachable and correct on HF Hub.

These tests stream the first few examples from each Hub repo without
downloading the whole dataset. They are skipped automatically if the Hub
is unreachable (no network / not logged in).
"""

from __future__ import annotations

import pytest
import numpy as np


def _hub_available(repo_id: str) -> bool:
    try:
        from huggingface_hub import dataset_info
        dataset_info(repo_id)
        return True
    except Exception:
        return False


URBANSOUND_AVAILABLE = _hub_available("i4ds/urbansound")
FREESOUNDS_AVAILABLE = _hub_available("i4ds/free_sounds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_n(repo_id: str, n: int = 5) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train", streaming=True)
    return [ex for ex, _ in zip(ds, range(n))]


# ---------------------------------------------------------------------------
# i4ds/urbansound
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not URBANSOUND_AVAILABLE, reason="i4ds/urbansound not reachable")
class TestUrbansoundHub:

    @pytest.fixture(scope="class")
    def examples(self):
        return _stream_n("i4ds/urbansound", n=10)

    def test_has_expected_columns(self, examples):
        required = {"audio", "text", "language", "prompt"}
        assert required.issubset(examples[0].keys()), \
            f"Missing columns: {required - examples[0].keys()}"

    def test_text_is_empty_silence(self, examples):
        non_empty = [e["text"] for e in examples if e["text"].strip()]
        assert not non_empty, f"Found non-empty text: {non_empty}"

    def test_language_is_english(self, examples):
        langs = {e["language"] for e in examples}
        assert langs == {"en"}, f"Unexpected languages: {langs}"

    def test_audio_is_16khz(self, examples):
        for ex in examples:
            assert ex["audio"]["sampling_rate"] == 16000

    def test_audio_duration_close_to_30s(self, examples):
        for ex in examples:
            arr = ex["audio"]["array"]
            sr  = ex["audio"]["sampling_rate"]
            dur = len(arr) / sr
            assert dur <= 30.5, f"Segment too long: {dur:.2f}s"
            assert dur >= 0.1,  f"Segment too short: {dur:.2f}s"

    def test_audio_has_non_zero_rms(self, examples):
        """Real background audio — should not be digital silence."""
        for ex in examples:
            arr = np.array(ex["audio"]["array"])
            rms = np.sqrt(np.mean(arr ** 2))
            assert rms > 1e-4, f"Audio looks like dead silence (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# i4ds/free_sounds
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FREESOUNDS_AVAILABLE, reason="i4ds/free_sounds not reachable")
class TestFreeSoundsHub:

    @pytest.fixture(scope="class")
    def examples(self):
        return _stream_n("i4ds/free_sounds", n=5)

    def test_has_expected_columns(self, examples):
        required = {"audio", "text", "language", "prompt"}
        assert required.issubset(examples[0].keys())

    def test_text_is_empty_silence(self, examples):
        non_empty = [e["text"] for e in examples if e["text"].strip()]
        assert not non_empty, f"Found non-empty text: {non_empty}"

    def test_language_is_english(self, examples):
        langs = {e["language"] for e in examples}
        assert langs == {"en"}, f"Unexpected languages: {langs}"

    def test_audio_is_16khz(self, examples):
        for ex in examples:
            assert ex["audio"]["sampling_rate"] == 16000

    def test_audio_duration_at_most_30s(self, examples):
        for ex in examples:
            arr = ex["audio"]["array"]
            dur = len(arr) / ex["audio"]["sampling_rate"]
            assert dur <= 30.5, f"Segment too long: {dur:.2f}s"
            assert dur >= 0.1,  f"Segment too short: {dur:.2f}s"
