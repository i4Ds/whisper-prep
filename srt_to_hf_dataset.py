#!/usr/bin/env python3
"""
Generate a Hugging Face dataset from folders of .srt and .mp3 files organized by series.

This script traverses an input directory containing subdirectories for each series (or movie),
loads each .srt file and its matching .mp3 audio file, and creates a dataset with the following fields:
  - audio: the path to the .mp3 audio file (cast to Audio feature)
  - sentence: the subtitle text without timestamps (concatenated cues)
  - series: the name of the series (subdirectory name)
  - srt: the raw .srt file content (including timestamps)
  - language: the language code (e.g., 'de')

The resulting dataset is saved to disk in Hugging Face format and can optionally be pushed to the Hub.
"""
import argparse
import json
from pathlib import Path

import pysubs2
from datasets import Audio, Dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Hugging Face dataset from .srt and .mp3 files by series"
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the directory containing series subfolders with .srt and .mp3 files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory where the Hugging Face dataset will be saved",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language code for all examples (default: de)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target audio sampling rate for the dataset (default: 16000)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Name of the dataset split (default: train)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional Hugging Face repo identifier to push the dataset to the Hub",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    # Traverse each series subfolder
    for series_dir in sorted(args.input_folder.iterdir()):
        if not series_dir.is_dir():
            continue
        series = series_dir.name
        for srt_path in sorted(series_dir.glob("*.srt")):
            audio_path = srt_path.with_suffix(".mp3")
            if not audio_path.exists():
                raise FileNotFoundError(f"No audio file found for {srt_path}")
            # Load raw SRT content
            srt_text = srt_path.read_text(encoding="utf-8")
            # Concatenate subtitle cues into a single sentence without timestamps
            subs = pysubs2.load(str(srt_path))
            sentence = " ".join(cue.text for cue in subs)

            records.append(
                {
                    "audio": str(audio_path),
                    "sentence": sentence,
                    "series": series,
                    "srt": srt_text,
                    "language": args.language,
                }
            )

    # Build Hugging Face dataset
    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
    ds_dict = DatasetDict({args.split_name: dataset})

    # Save to disk
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(args.output_dir))

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        ds_dict.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()