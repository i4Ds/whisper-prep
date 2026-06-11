from pathlib import Path

import pytest

from whisper_prep import _resolve_input_sources


def test_resolve_input_sources_rejects_direct_hf(tmp_path):
    with pytest.raises(ValueError, match="whisper_prep_download_hf"):
        _resolve_input_sources(
            {"hu_datasets": ["user/dataset"]},
            tmp_path / "audios",
            tmp_path / "transcripts",
        )


def test_resolve_input_sources_accepts_local_srt_folders(tmp_path):
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"

    resolved = _resolve_input_sources(
        {
            "source_audio_dir": str(audio_dir),
            "source_transcript_dir": str(transcript_dir),
        },
        tmp_path / "generated-audios",
        tmp_path / "generated-transcripts",
    )

    assert resolved == (audio_dir, transcript_dir, None, False)


def test_resolve_input_sources_rejects_multiple_routes(tmp_path):
    with pytest.raises(ValueError, match="exactly one input route"):
        _resolve_input_sources(
            {
                "transcripts_tsv": "mapping.tsv",
                "tsv_paths": ["sentences.tsv"],
                "clips_folders": ["clips"],
            },
            tmp_path / "audios",
            tmp_path / "transcripts",
        )
