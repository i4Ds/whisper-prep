"""Tests for ensure_extracted and use_source_audio_for_empty_full_segments."""

from __future__ import annotations

import csv
import shutil
import struct
import tarfile
import tempfile
import wave
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata_csv(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "UrbanSound8K.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slice_file_name", "fsID", "start", "end",
                        "salience", "fold", "classID", "class"],
            delimiter=",",
        )
        writer.writeheader()
        writer.writerow(dict(
            slice_file_name="fake.wav", fsID=1, start=0.0, end=2.0,
            salience=1, fold=1, classID=0, **{"class": "air_conditioner"},
        ))


def _make_silent_wav(path: Path, dur_s: float = 2.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(dur_s * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def _make_fake_archive(root: Path) -> Path:
    """Create a minimal UrbanSound8K.tar.gz with metadata only."""
    extracted = root / "UrbanSound8K"
    meta_dir = extracted / "metadata"
    _make_metadata_csv(meta_dir)

    archive = root / "UrbanSound8K.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(extracted, arcname="UrbanSound8K")

    shutil.rmtree(extracted)
    return archive


# ---------------------------------------------------------------------------
# Tests: ensure_extracted
# ---------------------------------------------------------------------------

class TestEnsureExtracted:

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_returns_extracted_path_when_already_done(self):
        """If metadata CSV already exists, extraction is skipped entirely."""
        from scripts.prepare_urbansound_silence_dataset import ensure_extracted

        # Pre-create the metadata — simulates a prior run
        meta_dir = self.tmp / "UrbanSound8K" / "metadata"
        _make_metadata_csv(meta_dir)

        result = ensure_extracted(self.tmp)
        assert result == self.tmp / "UrbanSound8K"

    def test_extracts_archive_when_needed(self):
        """When metadata is absent but archive exists, it is extracted."""
        from scripts.prepare_urbansound_silence_dataset import ensure_extracted

        _make_fake_archive(self.tmp)

        result = ensure_extracted(self.tmp)
        assert result == self.tmp / "UrbanSound8K"
        assert (result / "metadata" / "UrbanSound8K.csv").exists()

    def test_raises_when_archive_missing(self):
        """If neither metadata nor archive exist, FileNotFoundError is raised."""
        from scripts.prepare_urbansound_silence_dataset import ensure_extracted

        with pytest.raises(FileNotFoundError, match="Missing archive"):
            ensure_extracted(self.tmp)

    def test_no_extraction_when_metadata_exists_even_if_archive_present(self):
        """Archive is ignored when metadata is already there (idempotent)."""
        from scripts.prepare_urbansound_silence_dataset import ensure_extracted

        # Put both metadata AND archive
        _make_fake_archive(self.tmp)
        meta_dir = self.tmp / "UrbanSound8K" / "metadata"
        _make_metadata_csv(meta_dir)

        # Should not raise and should not delete/re-extract
        result = ensure_extracted(self.tmp)
        assert (result / "metadata" / "UrbanSound8K.csv").exists()


# ---------------------------------------------------------------------------
# Tests: use_source_audio_for_empty_full_segments (TSV path)
# ---------------------------------------------------------------------------

class TestUseSourceAudioForEmptyFullSegments:
    """Verify the optimisation that reuses the original clip path for short empty clips."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _run_dp(self, audio_path: Path, dur_s: float, use_source: bool) -> list[dict]:
        """Run DataProcessor in TSV mode with one clip, return parsed records."""
        import json
        from whisper_prep.generation.data_processor import DataProcessor

        srt_path = self.tmp / f"{audio_path.stem}.srt"
        srt_path.write_text("", encoding="utf-8")

        tsv_path = self.tmp / "transcripts.tsv"
        with tsv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "audio_path", "srt_path", "language", "duration_seconds"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(dict(
                id=audio_path.stem,
                audio_path=str(audio_path),
                srt_path=str(srt_path),
                language="en",
                duration_seconds=f"{dur_s:.6f}",
            ))

        output = self.tmp / "data.ljson"
        dump = self.tmp / "dump"
        dp = DataProcessor(
            audio_dir=str(self.tmp),
            transcript_dir=str(self.tmp),
            language="en",
            output=str(output),
            dump_dir=str(dump),
            transcripts_tsv=str(tsv_path),
            keep_empty_chance=1.0,
            use_source_audio_for_empty_full_segments=use_source,
        )
        dp.run()

        records = []
        with output.open() as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def test_source_path_reused_when_flag_true_and_clip_short(self):
        """Short clip (≤30 s) + empty SRT + flag=True → record points to original file."""
        audio = self.tmp / "short_clip.wav"
        _make_silent_wav(audio, dur_s=5.0)

        records = self._run_dp(audio, dur_s=5.0, use_source=True)

        assert len(records) == 1
        assert records[0]["audio_path"] == str(audio.absolute())
        assert records[0]["text"] == ""

    def test_new_segment_created_when_flag_false(self):
        """With flag=False a new segment file is always created in dump/."""
        audio = self.tmp / "short_clip2.wav"
        _make_silent_wav(audio, dur_s=5.0)

        records = self._run_dp(audio, dur_s=5.0, use_source=False)

        assert len(records) >= 1
        # With flag off, the path should point inside dump/, not to the original
        assert records[0]["audio_path"] != str(audio.absolute())
        assert Path(records[0]["audio_path"]).exists()

    def test_source_not_reused_when_clip_longer_than_30s(self):
        """Clips longer than 30 s must always be split, not reused directly."""
        audio = self.tmp / "long_clip.wav"
        _make_silent_wav(audio, dur_s=35.0)

        records = self._run_dp(audio, dur_s=35.0, use_source=True)

        # Should produce ≥2 segments, none pointing to the original file
        assert len(records) >= 2
        for r in records:
            assert r["audio_path"] != str(audio.absolute())


# ---------------------------------------------------------------------------
# Tests: freesound AUDIO_EXTS filtering
# ---------------------------------------------------------------------------

class TestAudioExtsFilter:

    def test_known_extensions_included(self):
        from scripts.prepare_freesound_dataset import AUDIO_EXTS
        for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
            assert ext in AUDIO_EXTS, f"{ext} should be in AUDIO_EXTS"

    def test_non_audio_extensions_excluded(self):
        from scripts.prepare_freesound_dataset import AUDIO_EXTS
        for ext in [".txt", ".csv", ".json", ".srt", ".pdf"]:
            assert ext not in AUDIO_EXTS, f"{ext} should NOT be in AUDIO_EXTS"

    def test_script_filters_correctly(self):
        """Only audio files in free_sound_dir should be picked up."""
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "audio"
            d.mkdir()
            _make_silent_wav(d / "clip.wav")
            (d / "readme.txt").write_text("ignore me")
            (d / "notes.csv").write_text("also ignore")

            from scripts.prepare_freesound_dataset import AUDIO_EXTS
            found = [f for f in sorted(d.iterdir()) if f.suffix.lower() in AUDIO_EXTS]
            assert len(found) == 1
            assert found[0].name == "clip.wav"
        finally:
            shutil.rmtree(tmp)
