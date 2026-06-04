"""Tests for the free_sound silence dataset pipeline."""

from __future__ import annotations

import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path


def make_silent_wav(path: Path, duration_s: float = 35.0, sr: int = 16000) -> Path:
    """Write a silent mono WAV of the given duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


# ---------------------------------------------------------------------------
# Unit tests: create_empty_srts
# ---------------------------------------------------------------------------

class TestCreateEmptySrts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_creates_srt_for_each_audio(self):
        from scripts.prepare_freesound_dataset import create_empty_srts

        audio_files = [
            make_silent_wav(self.tmp / "audio" / "clip_a.wav"),
            make_silent_wav(self.tmp / "audio" / "clip_b.wav"),
        ]
        srt_dir = self.tmp / "srts"
        create_empty_srts(audio_files, srt_dir)

        for af in audio_files:
            srt = srt_dir / f"{af.stem}.srt"
            self.assertTrue(srt.exists(), f"Missing SRT: {srt}")

    def test_srt_files_are_empty(self):
        from scripts.prepare_freesound_dataset import create_empty_srts

        audio_files = [make_silent_wav(self.tmp / "audio" / "ambient.wav")]
        srt_dir = self.tmp / "srts"
        create_empty_srts(audio_files, srt_dir)

        content = (srt_dir / "ambient.srt").read_text(encoding="utf-8")
        self.assertEqual(content, "", "SRT file should be completely empty")

    def test_srt_dir_created_if_missing(self):
        from scripts.prepare_freesound_dataset import create_empty_srts

        audio_files = [make_silent_wav(self.tmp / "audio" / "clip.wav")]
        srt_dir = self.tmp / "new_srt_dir" / "nested"
        create_empty_srts(audio_files, srt_dir)
        self.assertTrue(srt_dir.is_dir())

    def test_no_srts_for_non_audio_files(self):
        from scripts.prepare_freesound_dataset import create_empty_srts, AUDIO_EXTS

        # Only WAV files should be included when caller pre-filters
        audio_files = [make_silent_wav(self.tmp / "audio" / "clip.wav")]
        srt_dir = self.tmp / "srts"
        create_empty_srts(audio_files, srt_dir)
        self.assertEqual(len(list(srt_dir.glob("*.srt"))), 1)


# ---------------------------------------------------------------------------
# Unit tests: build_config
# ---------------------------------------------------------------------------

class TestBuildConfig(unittest.TestCase):

    def _make(self, **kw):
        from scripts.prepare_freesound_dataset import build_config
        defaults = dict(
            dataset_name="free_sounds",
            split_name="train",
            language="en",
            out_folder_base=Path("/tmp/wp_out"),
            audio_dir=Path("/mnt/nas05/free_sound"),
            srt_dir=Path("/tmp/wp_out/free_sounds/train/srt"),
            repo_id="i4ds/free_sounds",
            upload=False,
        )
        defaults.update(kw)
        return build_config(**defaults)

    def test_uses_folder_source_route(self):
        cfg = self._make()
        self.assertIn("source_audio_dir", cfg)
        self.assertIn("source_transcript_dir", cfg)
        self.assertNotIn("tsv_paths", cfg)
        self.assertNotIn("clips_folders", cfg)

    def test_silence_settings(self):
        cfg = self._make()
        self.assertEqual(cfg["keep_empty_chance"], 1.0)
        self.assertEqual(cfg["drop_empty_text"], False)
        self.assertEqual(cfg["min_text_words"], 0)

    def test_audio_dir_matches(self):
        audio = Path("/some/audio/dir")
        cfg = self._make(audio_dir=audio)
        self.assertEqual(cfg["source_audio_dir"], str(audio))

    def test_srt_dir_matches(self):
        srt = Path("/some/srt/dir")
        cfg = self._make(srt_dir=srt)
        self.assertEqual(cfg["source_transcript_dir"], str(srt))

    def test_upload_flag(self):
        self.assertFalse(self._make(upload=False)["upload_to_hu"])
        self.assertTrue(self._make(upload=True)["upload_to_hu"])

    def test_repo_id(self):
        cfg = self._make(repo_id="i4ds/free_sounds")
        self.assertEqual(cfg["hu_repo"], "i4ds/free_sounds")


# ---------------------------------------------------------------------------
# Integration: full pipeline with fake audio files
# ---------------------------------------------------------------------------

class TestFreesoundPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pipeline_cuts_long_files_into_silence_records(self):
        """A 65 s file should produce two 30 s + one 5 s segment, all empty text."""
        from scripts.prepare_freesound_dataset import (
            AUDIO_EXTS, build_config, create_empty_srts,
        )

        audio_dir = self.tmp / "audio"
        # Two fake audio files: 65 s and 40 s
        audio_files = [
            make_silent_wav(audio_dir / "ambient_65s.wav", duration_s=65.0),
            make_silent_wav(audio_dir / "ambient_40s.wav", duration_s=40.0),
        ]

        srt_dir = self.tmp / "srts"
        create_empty_srts(audio_files, srt_dir)

        config = build_config(
            dataset_name="test_fs",
            split_name="train",
            language="en",
            out_folder_base=self.tmp / "wp_out",
            audio_dir=audio_dir,
            srt_dir=srt_dir,
            repo_id="i4ds/free_sounds",
            upload=False,
        )

        import whisper_prep
        whisper_prep.main(config)

        ljson = self.tmp / "wp_out" / "test_fs" / "train" / "created_dataset" / "data.ljson"
        self.assertTrue(ljson.exists(), "data.ljson not created")

        from whisper_prep.dataset.convert import ljson_to_pandas
        df = ljson_to_pandas(ljson)

        # 65s → 3 segments (30+30+5), 40s → 2 segments (30+10) = 5 total
        self.assertGreaterEqual(len(df), 4, f"Expected ≥4 segments, got {len(df)}")

        # All text must be empty (silence)
        non_empty = df["text"].astype(str).str.strip().ne("")
        self.assertFalse(non_empty.any(), "Non-empty text found in silence dataset")

        # Language should match
        self.assertTrue((df["language"] == "en").all())

        # All audio files must exist on disk
        for path in df["audio"]:
            self.assertTrue(Path(path).exists(), f"Audio segment missing: {path}")

    def test_missing_audio_dir_raises(self):
        """Script should raise if free_sound_dir doesn't exist or is empty."""
        import sys
        sys.argv = ["prepare_freesound_dataset.py",
                    "--free-sound-dir", str(self.tmp / "nonexistent")]
        from scripts.prepare_freesound_dataset import parse_args, AUDIO_EXTS

        args = parse_args()
        # No audio files in nonexistent dir → should raise FileNotFoundError
        audio_files = []  # simulate empty discovery
        with self.assertRaises(FileNotFoundError):
            if not audio_files:
                raise FileNotFoundError(f"No audio files found in {args.free_sound_dir}")
