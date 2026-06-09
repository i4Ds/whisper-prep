from pathlib import Path

from whisper_prep.audio.vad import silero_speech_ratio
from whisper_prep.generation.data_processor import DataProcessor


ASSETS = Path("tests/assets/empty_vad")


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
