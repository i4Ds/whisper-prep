#!/usr/bin/env python3
"""
Download Hugging Face ASR data into the local whisper-prep folder layout.
"""

import argparse
from pathlib import Path

from whisper_prep.utils import save_hu_dataset_locally


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets as local audio plus SRTs or sentence TSV."
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Hugging Face dataset ID. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder that will contain audios/, transcripts/, and TSV metadata.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face input split to download (default: train).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "srt", "sentences"],
        default="auto",
        help="Download mode: auto prefers SRTs, srt requires SRTs, sentences writes hf_sentences.tsv.",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Fallback language code for transcripts_mapping.tsv when the dataset has no language column.",
    )
    parser.add_argument(
        "--episode-ids-file",
        type=Path,
        default=None,
        help="Optional newline-separated IDs to keep. Requires the dataset to have an id column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = args.output_dir / "audios"
    transcript_dir = args.output_dir / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "out_folder": args.output_dir,
        "split_name": args.split,
        "hu_input_split": args.split,
        "hu_datasets": args.datasets,
        "hf_input_format": args.format,
        "language": args.language,
    }
    if args.episode_ids_file:
        config["hu_episode_ids_file"] = str(args.episode_ids_file)

    downloaded = save_hu_dataset_locally(config, audio_dir, transcript_dir)
    if downloaded.sentence_tsvs:
        print("Downloaded sentence-level data:")
        for path in downloaded.sentence_tsvs:
            print(f"  {path}")
    if downloaded.transcripts_tsv:
        print("Downloaded SRT-level data:")
        print(f"  {downloaded.transcripts_tsv}")


if __name__ == "__main__":
    main()
