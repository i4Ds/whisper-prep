import csv
import sys
from pathlib import Path

import numpy as np

from whisper_prep import utils
from whisper_prep.download_hf_dataset import main as download_main


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def select_columns(self, columns):
        return FakeDataset([{column: row[column] for column in columns} for row in self.rows])

    def filter(self, predicate):
        return FakeDataset([row for row in self.rows if predicate(row)])


def fake_audio():
    return {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}


def patch_hf_dataset(monkeypatch, rows):
    monkeypatch.setattr(utils, "load_dataset", lambda name, split: FakeDataset(rows))
    monkeypatch.setattr(
        utils,
        "concatenate_datasets",
        lambda datasets: FakeDataset(
            [row for dataset in datasets for row in dataset.rows]
        ),
    )


def read_tsv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_save_hu_dataset_locally_writes_sentence_tsv(tmp_path, monkeypatch):
    patch_hf_dataset(
        monkeypatch,
        [
            {
                "id": "sample/1",
                "audio": fake_audio(),
                "sentence": "Hallo Welt",
                "client_id": "speaker-1",
            }
        ],
    )

    audio_dir = tmp_path / "audios"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir()
    transcript_dir.mkdir()

    downloaded = utils.save_hu_dataset_locally(
        {
            "out_folder": tmp_path,
            "split_name": "train",
            "hu_datasets": ["fake/dataset"],
            "hf_input_format": "sentences",
        },
        audio_dir,
        transcript_dir,
    )

    assert downloaded.sentence_tsvs == [str(tmp_path / "hf_sentences.tsv")]
    assert downloaded.transcripts_tsv is None
    rows = read_tsv(tmp_path / "hf_sentences.tsv")
    assert rows == [
        {
            "path": "sample_1.wav",
            "sentence": "Hallo Welt",
            "client_id": "speaker-1",
        }
    ]
    assert (audio_dir / "sample_1.wav").exists()
    assert not (tmp_path / "transcripts_mapping.tsv").exists()


def test_save_hu_dataset_locally_writes_srt_mapping(tmp_path, monkeypatch):
    srt_text = "1\n00:00:00,000 --> 00:00:01,000\nHallo Welt\n"
    patch_hf_dataset(
        monkeypatch,
        [
            {
                "id": "episode-1",
                "audio": fake_audio(),
                "srt": srt_text,
                "language": "de",
            }
        ],
    )

    audio_dir = tmp_path / "audios"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir()
    transcript_dir.mkdir()

    downloaded = utils.save_hu_dataset_locally(
        {
            "out_folder": tmp_path,
            "split_name": "train",
            "hu_datasets": ["fake/dataset"],
            "hf_input_format": "srt",
        },
        audio_dir,
        transcript_dir,
    )

    assert downloaded.sentence_tsvs == []
    assert downloaded.transcripts_tsv == str(tmp_path / "transcripts_mapping.tsv")
    assert (audio_dir / "episode-1.wav").exists()
    assert (transcript_dir / "episode-1.srt").read_text(encoding="utf-8") == srt_text

    rows = read_tsv(tmp_path / "transcripts_mapping.tsv")
    assert rows == [
        {
            "srt_path": str(transcript_dir / "episode-1.srt"),
            "audio_path": str(audio_dir / "episode-1.wav"),
            "language": "de",
            "id": "episode-1",
        }
    ]


def test_download_cli_passes_expected_config(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_save(config, audio_dir, transcript_dir):
        calls.append((config, audio_dir, transcript_dir))
        return utils.DownloadedDatasetPaths(
            [str(config["out_folder"] / "hf_sentences.tsv")]
        )

    monkeypatch.setattr(
        "whisper_prep.download_hf_dataset.save_hu_dataset_locally", fake_save
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_hf_dataset.py",
            "--dataset",
            "one/dataset",
            "--dataset",
            "two/dataset",
            "--output-dir",
            str(tmp_path),
            "--split",
            "validation",
            "--format",
            "sentences",
            "--language",
            "de",
        ],
    )

    download_main()

    assert len(calls) == 1
    config, audio_dir, transcript_dir = calls[0]
    assert config["hu_datasets"] == ["one/dataset", "two/dataset"]
    assert config["hu_input_split"] == "validation"
    assert config["hf_input_format"] == "sentences"
    assert config["language"] == "de"
    assert audio_dir == tmp_path / "audios"
    assert transcript_dir == tmp_path / "transcripts"
    assert audio_dir.exists()
    assert transcript_dir.exists()
    assert "Downloaded sentence-level data" in capsys.readouterr().out
