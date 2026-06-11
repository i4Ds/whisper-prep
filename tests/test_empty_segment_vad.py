import csv
import re
from pathlib import Path

import json
import torch
import torchaudio

from whisper_prep.audio.vad import silero_speech_ratio
from whisper_prep.generation.data_processor import DataProcessor, SAMPLE_RATE


ASSETS = Path("tests/assets/empty_vad")
PUNCTUATION_HALLUCINATION_ASSETS = Path("tests/assets/punctuation_hallucination")


def test_silero_speech_ratio_separates_empty_text_examples():
    assert silero_speech_ratio(str(ASSETS / "empty_with_speech.mp3")) > 0.06
    assert silero_speech_ratio(str(ASSETS / "empty_without_speech.mp3")) > 0.06
    assert silero_speech_ratio(str(ASSETS / "empty_without_speech_2.mp3")) <= 0.06


def test_empty_segment_vad_gate_removes_speechy_empty_audio(tmp_path):
    speechy = tmp_path / "speechy.mp3"
    quiet = tmp_path / "quiet.mp3"
    speechy.write_bytes((ASSETS / "empty_with_speech.mp3").read_bytes())
    quiet.write_bytes((ASSETS / "empty_without_speech_2.mp3").read_bytes())

    processor = DataProcessor(
        audio_dir=str(tmp_path),
        transcript_dir=str(tmp_path),
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        validate_empty_with_vad=True,
        empty_vad_max_speech_ratio=0.06,
    )

    assert not processor._empty_segment_passes_vad(str(speechy))
    assert not speechy.exists()
    assert processor._empty_segment_passes_vad(str(quiet))
    assert quiet.exists()


def test_empty_segment_vad_advances_after_rejecting_speechy_gap(tmp_path):
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir()
    transcript_dir.mkdir()

    speechy_gap, sr = torchaudio.load(ASSETS / "empty_with_speech.mp3")
    speechy_gap = speechy_gap.mean(dim=0)
    if sr != SAMPLE_RATE:
        speechy_gap = torchaudio.functional.resample(speechy_gap, sr, SAMPLE_RATE)

    prefix = torch.zeros(SAMPLE_RATE)
    suffix = torch.zeros(SAMPLE_RATE * 8)
    audio = torch.cat([prefix, speechy_gap, suffix]).unsqueeze(0)
    audio_path = audio_dir / "sample.wav"
    torchaudio.save(audio_path, audio, SAMPLE_RATE)

    gap_start_ms = 1000
    next_caption_start_ms = 32000
    transcript_dir.joinpath("sample.srt").write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:01,000",
                "Before",
                "",
                "2",
                f"00:00:{next_caption_start_ms // 1000:02d},{next_caption_start_ms % 1000:03d} --> 00:00:{(next_caption_start_ms + 1000) // 1000:02d},{(next_caption_start_ms + 1000) % 1000:03d}",
                "After",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    processor = DataProcessor(
        audio_dir=str(audio_dir),
        transcript_dir=str(transcript_dir),
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        keep_empty_chance=1.0,
        validate_empty_with_vad=True,
        empty_vad_max_speech_ratio=0.06,
    )
    processor.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "data.ljson").read_text(encoding="utf-8").splitlines()
    ]
    names = [Path(record["audio_path"]).name for record in records]

    assert f"{gap_start_ms}.mp3" not in names
    assert not (tmp_path / "dump" / "sample" / f"{gap_start_ms}.mp3").exists()
    assert any("After" in record["text"] for record in records)


def test_long_real_silence_is_split_and_kept_with_empty_vad(tmp_path):
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir()
    transcript_dir.mkdir()

    audio_path = audio_dir / "sample.wav"
    audio = torch.zeros(1, SAMPLE_RATE * 75)
    torchaudio.save(audio_path, audio, SAMPLE_RATE)
    (transcript_dir / "sample.srt").write_text("", encoding="utf-8")

    processor = DataProcessor(
        audio_dir=str(audio_dir),
        transcript_dir=str(transcript_dir),
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        keep_empty_chance=1.0,
        validate_empty_with_vad=True,
        empty_vad_max_speech_ratio=0.06,
    )
    processor.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "data.ljson").read_text(encoding="utf-8").splitlines()
    ]
    names = [Path(record["audio_path"]).name for record in records]

    assert names == ["0.mp3", "30000.mp3", "60000.mp3"]
    assert all(record["text"] == "" for record in records)
    assert all(Path(record["audio_path"]).exists() for record in records)


def test_real_slice_drops_punctuation_hallucination_and_keeps_timestamps(tmp_path):
    audio_path = PUNCTUATION_HALLUCINATION_ASSETS / "punctuation_hallucination_slice.mp3"
    srt_path = PUNCTUATION_HALLUCINATION_ASSETS / "punctuation_hallucination_slice.srt"
    tsv_path = tmp_path / "transcripts.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "audio_path", "srt_path", "language"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "punctuation_hallucination_real",
                "audio_path": str(audio_path),
                "srt_path": str(srt_path),
                "language": "fr",
            }
        )

    processor = DataProcessor(
        audio_dir="",
        transcript_dir="",
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        transcripts_tsv=str(tsv_path),
        cut_initial_audio=True,
        keep_empty_chance=1.0,
        validate_empty_with_vad=True,
        empty_vad_max_speech_ratio=0.06,
    )
    processor.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "data.ljson").read_text(encoding="utf-8").splitlines()
    ]
    names = [Path(record["audio_path"]).name for record in records]
    combined_text = "\n".join(record["text"] for record in records)

    assert records
    assert any("musique acophonique" in record["text"] for record in records)
    assert "............" not in combined_text
    assert ".............................." not in combined_text
    assert all(not re.fullmatch(r"(?:<\|[0-9.]+\|>|\s|\.)+", record["text"]) for record in records)

    for record in records:
        audio, sample_rate = torchaudio.load(record["audio_path"])
        duration_ms = round(audio.size(1) * 1000 / sample_rate)
        timestamps = [
            round(float(value) * 1000)
            for value in re.findall(r"<\|([0-9]+\.[0-9]+)\|>", record["text"])
        ]
        assert all(0 <= timestamp <= duration_ms for timestamp in timestamps)

    report = tmp_path / "filtered_punctuation_hallucination_examples.csv"
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "............" in report_text
    assert "punctuation_hallucination" in report_text
