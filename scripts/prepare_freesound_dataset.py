#!/usr/bin/env python3
"""Split free_sound ambient files into ≤30-s silence segments via whisper-prep.

The files are already long enough — no concatenation needed. The script:
  1. Creates an empty .srt for every audio file (no speech → pure silence).
  2. Writes a whisper-prep config using the folder-source route.
  3. Runs whisper_prep.main() → DataProcessor cuts each file into ≤30 s chunks
     (keep_empty_chance=1.0 so every segment is kept).
  4. Saves a HuggingFace dataset and optionally pushes to the Hub.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--free-sound-dir",
        type=Path,
        default=Path("/mnt/nas05/data01/vincenzo/UrbanSound8k/free_sound"),
    )
    p.add_argument(
        "--out-folder-base",
        type=Path,
        default=Path("/mnt/nas05/data01/vincenzo/UrbanSound8k/whisper_prep"),
    )
    p.add_argument("--dataset-name", default="free_sounds")
    p.add_argument("--split-name", default="train")
    p.add_argument("--language", default="en")
    p.add_argument("--repo-id", default="i4ds/free_sounds")
    p.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/freesound_train.yaml"),
    )
    p.add_argument("--upload", action="store_true", help="Push to HuggingFace Hub.")
    return p.parse_args()


def create_empty_srts(audio_files: list[Path], srt_dir: Path) -> None:
    """Write an empty .srt file for each audio file."""
    srt_dir.mkdir(parents=True, exist_ok=True)
    for audio in audio_files:
        (srt_dir / f"{audio.stem}.srt").write_text("", encoding="utf-8")
    print(f"Created {len(audio_files)} empty SRT files in {srt_dir}")


def build_config(
    *,
    dataset_name: str,
    split_name: str,
    language: str,
    out_folder_base: Path,
    audio_dir: Path,
    srt_dir: Path,
    repo_id: str,
    upload: bool,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "language": language,
        "out_folder_base": str(out_folder_base),
        # Folder-source route: DataProcessor reads audio + SRT directly
        "source_audio_dir": str(audio_dir),
        "source_transcript_dir": str(srt_dir),
        # DataProcessor settings
        "keep_empty_chance": 1.0,
        "drop_empty_text": False,
        "min_text_words": 0,
        "netflix_normalize": False,
        "cut_initial_audio": False,
        "filter_french": False,
        "filter_english": False,
        "filter_words": [],
        # Upload
        "upload_to_hu": upload,
        "hu_repo": repo_id,
        "hu_private": False,
    }


def main() -> None:
    args = parse_args()

    audio_files = sorted(
        f for f in args.free_sound_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {args.free_sound_dir}")

    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  {f.name}")

    out_folder = args.out_folder_base / args.dataset_name / args.split_name
    srt_dir = out_folder / "srt"

    create_empty_srts(audio_files, srt_dir)

    config = build_config(
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        language=args.language,
        out_folder_base=args.out_folder_base,
        audio_dir=args.free_sound_dir,
        srt_dir=srt_dir,
        repo_id=args.repo_id,
        upload=args.upload,
    )

    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote config to {args.config_path}")

    # Clean up previous DataProcessor output so it doesn't raise on re-run
    for subdir in ["created_dataset", "hf"]:
        d = out_folder / subdir
        if d.exists():
            shutil.rmtree(d)

    import whisper_prep
    whisper_prep.main(config)


if __name__ == "__main__":
    main()
