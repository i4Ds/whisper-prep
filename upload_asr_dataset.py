#!/usr/bin/env python3
"""
Convert a TSV file of audio metadata into a Hugging Face ASR dataset and push to the Hub.

Usage:
  python upload_asr_dataset.py \
      --tsv PATH/TO/DATA.tsv \
      --repo_id USERNAME/DATASET_NAME \
      [--split SPLIT_NAME] [--sampling_rate RATE]
"""

import argparse

import pandas as pd
from datasets import Audio, Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(
        description="Convert TSV to Hugging Face ASR dataset and push to the Hub"
    )
    parser.add_argument(
        "--tsv", required=True, help="Path to the TSV file containing the metadata"
    )
    parser.add_argument(
        "--repo_id", required=True,
        help="Hugging Face dataset repo ID (e.g., username/dataset_name)"
    )
    parser.add_argument(
        "--split", default="train",
        help="Name of the dataset split (default: train)"
    )
    parser.add_argument(
        "--audio_column", default="path",
        help="TSV column name for audio file paths (default: path)"
    )
    parser.add_argument(
        "--text_column", default="sentence",
        help="TSV column name for transcriptions (default: sentence)"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=16000,
        help="Sampling rate for audio casting (default: 16000)"
    )
    args = parser.parse_args()

    # Load the TSV into a DataFrame
    df = pd.read_csv(args.tsv, sep="\t")

    # Rename columns to match Hugging Face ASR dataset format
    df = df.rename(
        columns={args.audio_column: "audio", args.text_column: "text"}
    )

    # Create a Dataset and cast the audio column
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    # Wrap in a DatasetDict under the specified split
    dataset = DatasetDict({args.split: ds})

    # Push to Hugging Face Hub (requires 'huggingface-cli login')
    dataset.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()