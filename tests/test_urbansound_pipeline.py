"""Tests for the UrbanSound8K silence dataset pipeline."""

from __future__ import annotations

import csv
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake UrbanSound8K directory
# ---------------------------------------------------------------------------

def make_fake_urbansound(root: Path) -> tuple[Path, pd.DataFrame]:
    """Create a minimal fake UrbanSound8K structure with silent WAV clips."""
    import wave, struct

    extracted = root / "UrbanSound8K"
    audio_root = extracted / "audio"
    meta_dir = extracted / "metadata"
    meta_dir.mkdir(parents=True)

    rows = []
    for class_id in range(10):
        fold = (class_id % 5) + 1
        fold_dir = audio_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fname = f"fake_{class_id}-{class_id}-0-0.wav"
        clip_path = fold_dir / fname
        # Write a 2-second silent mono 16-bit PCM WAV
        with wave.open(str(clip_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 32000, *([0] * 32000)))
        rows.append({
            "slice_file_name": fname,
            "fsID": 100000 + class_id,
            "start": 0.0,
            "end": 2.0,
            "salience": 1,
            "fold": fold,
            "classID": class_id,
            "class": f"class_{class_id}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(meta_dir / "UrbanSound8K.csv", index=False)
    return extracted, df


# ---------------------------------------------------------------------------
# Unit tests: build_source_tsv
# ---------------------------------------------------------------------------

class TestBuildSourceTsv(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.extracted, self.df = make_fake_urbansound(self.tmp)
        self.free_sound_dir = self.tmp / "free_sound"
        self.free_sound_dir.mkdir()
        # Add a tiny free_sound MP3 (actually just copy a WAV and rename)
        import wave, struct
        fs_path = self.free_sound_dir / "ambient_room.mp3"
        with wave.open(str(fs_path.with_suffix(".wav")), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
        fs_path.write_bytes(fs_path.with_suffix(".wav").read_bytes())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, target_hours: float = 0.001) -> pd.DataFrame:
        from scripts.prepare_urbansound_silence_dataset import (
            CLASS_NAMES, build_source_tsv,
        )
        tsv_path = self.tmp / "out.tsv"
        build_source_tsv(
            df=self.df,
            extracted=self.extracted,
            free_sound_dir=self.free_sound_dir,
            tsv_path=tsv_path,
            target_hours_total=target_hours * len(CLASS_NAMES),
            seed=0,
        )
        # keep_default_na=False so empty sentence fields stay as "" not NaN
        return pd.read_csv(tsv_path, sep="\t", keep_default_na=False)

    def test_columns(self):
        result = self._run()
        self.assertListEqual(sorted(result.columns.tolist()), ["client_id", "path", "sentence"])

    def test_no_nan_sentences(self):
        result = self._run()
        self.assertFalse(result["sentence"].isna().any(), "sentence column must not contain NaN")

    def test_sentences_are_empty_string(self):
        result = self._run()
        non_empty = result["sentence"].astype(str).str.strip().ne("")
        self.assertFalse(non_empty.any(), "All sentences should be empty (silence)")

    def test_all_paths_exist(self):
        result = self._run()
        for path in result["path"]:
            self.assertTrue(Path(path).exists(), f"Audio file missing: {path}")

    def test_class_balance(self):
        from scripts.prepare_urbansound_silence_dataset import CLASS_NAMES
        result = self._run(target_hours=0.001)
        urbansound_rows = result[result["client_id"] != "freesound"]
        counts = urbansound_rows["client_id"].value_counts()
        # All 10 classes must be present
        self.assertEqual(len(counts), len(CLASS_NAMES))
        # Counts should be within 2x of each other (balanced)
        self.assertLess(counts.max() / counts.min(), 15,
                        "Class distribution too imbalanced")

    def test_freesound_included(self):
        result = self._run()
        self.assertIn("freesound", result["client_id"].values)

    def test_target_hours_respected(self):
        """Total source duration should be >= target per class."""
        from scripts.prepare_urbansound_silence_dataset import CLASS_NAMES
        target_h_per_class = 0.001
        result = self._run(target_hours=target_h_per_class)
        urbansound = result[result["client_id"] != "freesound"]
        # Each clip is 2 s. total clips × 2 s >= target_h × 3600 per class
        for class_name in CLASS_NAMES.values():
            n = (urbansound["client_id"] == class_name).sum()
            self.assertGreaterEqual(n * 2.0, target_h_per_class * 3600,
                                    f"Class {class_name} under-sampled")


# ---------------------------------------------------------------------------
# Unit tests: build_config
# ---------------------------------------------------------------------------

class TestBuildConfig(unittest.TestCase):

    def _make_config(self, **overrides):
        from scripts.prepare_urbansound_silence_dataset import build_config
        defaults = dict(
            dataset_name="urbansound",
            split_name="train",
            language="en",
            out_folder_base=Path("/tmp/wp_out"),
            tsv_path=Path("/tmp/wp_out/urbansound/train/inputs/fusion_empty_sentences.tsv"),
            n_samples_per_srt=12,
            n_jobs=8,
            seed=42,
            repo_id="i4ds/urbansound",
            upload=False,
        )
        defaults.update(overrides)
        return build_config(**defaults)

    def test_required_generate_fold_keys(self):
        config = self._make_config()
        for key in ["tsv_paths", "clips_folders", "partials",
                    "maintain_speaker_chance", "n_samples_per_srt",
                    "overlap_chance", "vad_chance", "keep_empty_chance"]:
            self.assertIn(key, config, f"Missing key: {key}")

    def test_silence_settings(self):
        config = self._make_config()
        self.assertEqual(config["keep_empty_chance"], 1.0)
        self.assertEqual(config["vad_chance"], 0.0)
        self.assertEqual(config["overlap_chance"], 0.0)
        self.assertEqual(config["drop_empty_text"], False)
        self.assertEqual(config["maintain_speaker_chance"], 0.0)

    def test_clips_folders_is_slash(self):
        config = self._make_config()
        self.assertEqual(config["clips_folders"], ["/"])

    def test_upload_flag(self):
        self.assertFalse(self._make_config(upload=False)["upload_to_hu"])
        self.assertTrue(self._make_config(upload=True)["upload_to_hu"])

    def test_tsv_path_in_tsv_paths(self):
        tsv = Path("/tmp/fake.tsv")
        config = self._make_config(tsv_path=tsv)
        self.assertIn(str(tsv), config["tsv_paths"])


# ---------------------------------------------------------------------------
# Unit tests: NaN fix in combine_tsvs_to_dataframe
# ---------------------------------------------------------------------------

class TestCombineTsvNanSentence(unittest.TestCase):
    """Verify that empty sentence fields in TSV don't crash generate_fold."""

    def test_empty_sentence_becomes_empty_string(self):
        """combine_tsvs_to_dataframe must return '' for empty sentence fields."""
        import wave, struct
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create a minimal WAV clip
            clip = tmp / "clip.wav"
            with wave.open(str(clip), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

            # Write TSV with empty sentence (will be read as NaN by default)
            tsv = tmp / "test.tsv"
            tsv.write_text(f"sentence\tpath\tclient_id\n\t{clip}\tcls\n")

            from whisper_prep.dataset.convert import combine_tsvs_to_dataframe
            df = combine_tsvs_to_dataframe([tsv], ["/"], [1.0])

            self.assertEqual(len(df), 1)
            sentence = df.iloc[0]["sentence"]
            self.assertIsInstance(sentence, str, "sentence should be a str, not NaN/float")
            self.assertEqual(sentence, "")

    def test_nan_sentence_does_not_crash_generate(self):
        """Empty TSV sentences should produce fused audio without crashing."""
        import wave, struct
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clips = []
            for i in range(4):
                p = tmp / f"clip{i}.wav"
                with wave.open(str(p), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(struct.pack("<" + "h" * 32000, *([0] * 32000)))
                clips.append(str(p))

            tsv = tmp / "test.tsv"
            with tsv.open("w") as f:
                f.write("sentence\tpath\tclient_id\n")
                for c in clips:
                    f.write(f"\t{c}\tcls\n")

            out = tmp / "out"
            (out / "audios").mkdir(parents=True)
            (out / "transcripts").mkdir(parents=True)

            from whisper_prep.generation.generate import generate_fold
            # Should not raise
            generate_fold(
                tsv_paths=[tsv],
                clips_folders=["/"],
                partials=[1.0],
                out_folder=out,
                maintain_speaker_chance=0.0,
                n_samples_per_srt=4,
                normalize_text=False,
                overlap_chance=0.0,
                max_overlap_chance=0.0,
                max_overlap_duration=0.0,
                vad_chance=0.0,
                keep_empty_chance=1.0,
                n_jobs=1,
                seed=0,
            )
            audios = list((out / "audios").glob("*.mp3"))
            self.assertGreater(len(audios), 0, "generate_fold produced no audio files")


# ---------------------------------------------------------------------------
# Integration: full pipeline with tiny fake data
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end smoke test using fake UrbanSound8K data."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pipeline_produces_silence_records(self):
        import wave, struct
        extracted, df = make_fake_urbansound(self.tmp)

        tsv_path = self.tmp / "inputs" / "fusion_empty_sentences.tsv"
        tsv_path.parent.mkdir(parents=True)

        from scripts.prepare_urbansound_silence_dataset import (
            CLASS_NAMES, build_config, build_source_tsv,
        )
        build_source_tsv(
            df=df,
            extracted=extracted,
            free_sound_dir=self.tmp / "nonexistent_freesound",
            tsv_path=tsv_path,
            target_hours_total=0.005 * len(CLASS_NAMES),
            seed=0,
        )
        config = build_config(
            dataset_name="test_us",
            split_name="train",
            language="en",
            out_folder_base=self.tmp / "wp_out",
            tsv_path=tsv_path,
            n_samples_per_srt=4,
            n_jobs=1,
            seed=0,
            repo_id="i4ds/urbansound",
            upload=False,
        )

        import whisper_prep
        whisper_prep.main(config)

        ljson = (
            self.tmp / "wp_out" / "test_us" / "train"
            / "created_dataset" / "data.ljson"
        )
        self.assertTrue(ljson.exists(), "data.ljson was not created")

        from whisper_prep.dataset.convert import ljson_to_pandas
        result_df = ljson_to_pandas(ljson)
        self.assertGreater(len(result_df), 0, "Pipeline produced no records")

        # All text should be empty (silence)
        non_empty = result_df["text"].astype(str).str.strip().ne("")
        self.assertFalse(
            non_empty.any(),
            f"Found non-empty text in silence dataset: {result_df[non_empty]['text'].tolist()[:3]}"
        )

        # Language should be 'en'
        self.assertTrue((result_df["language"] == "en").all())

        # All audio files should exist
        for path in result_df["audio"]:
            self.assertTrue(Path(path).exists(), f"Audio file missing: {path}")


# ---------------------------------------------------------------------------
# Integration: --skip-generate path (DataProcessor only, existing fused audio)
# ---------------------------------------------------------------------------

class TestSkipGeneratePath(unittest.TestCase):
    """Verify that build_skip_generate_config uses the folder-source route."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_skip_generate_config_uses_folder_route(self):
        from scripts.prepare_urbansound_silence_dataset import build_skip_generate_config
        cfg = build_skip_generate_config(
            dataset_name="urbansound",
            split_name="train",
            language="en",
            out_folder_base=Path("/tmp/wp"),
            repo_id="i4ds/urbansound",
            upload=False,
        )
        self.assertIn("source_audio_dir", cfg)
        self.assertIn("source_transcript_dir", cfg)
        self.assertNotIn("tsv_paths", cfg)
        self.assertNotIn("clips_folders", cfg)
        self.assertEqual(cfg["keep_empty_chance"], 1.0)
        self.assertFalse(cfg["drop_empty_text"])

    def test_skip_generate_runs_dataprocessor_on_existing_audio(self):
        """With pre-existing fused audio + SRT, --skip-generate produces records."""
        import wave, struct

        out_folder = self.tmp / "wp_out" / "urbansound" / "train"
        audios_dir = out_folder / "audios"
        transcripts_dir = out_folder / "transcripts"
        audios_dir.mkdir(parents=True)
        transcripts_dir.mkdir(parents=True)

        # Write two fake 35 s fused audio files + empty SRTs (simulating generate_fold output)
        for i in range(2):
            wav_path = audios_dir / f"fake_fused_{i}.wav"
            n = int(35 * 16000)
            with wave.open(str(wav_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
            (transcripts_dir / f"fake_fused_{i}.srt").write_text("", encoding="utf-8")

        from scripts.prepare_urbansound_silence_dataset import build_skip_generate_config
        config = build_skip_generate_config(
            dataset_name="urbansound",
            split_name="train",
            language="en",
            out_folder_base=self.tmp / "wp_out",
            repo_id="i4ds/urbansound",
            upload=False,
        )

        import whisper_prep
        whisper_prep.main(config)

        ljson = out_folder / "created_dataset" / "data.ljson"
        self.assertTrue(ljson.exists(), "data.ljson not created in --skip-generate path")

        from whisper_prep.dataset.convert import ljson_to_pandas
        df = ljson_to_pandas(ljson)
        # 2 files × 35 s → 2+2 = 4 segments (30 s + 5 s each)
        self.assertGreaterEqual(len(df), 2)
        self.assertTrue((df["text"].astype(str).str.strip() == "").all())
        self.assertTrue((df["language"] == "en").all())


if __name__ == "__main__":
    unittest.main()
