#!/usr/bin/env python3
"""Build a balanced UrbanSound8K silence dataset using whisper-prep.

Pipeline:
  1. Extract UrbanSound8K archive (if needed).
  2. Build a single source TSV: clips from all 10 classes (repeated to reach
     target hours) plus free_sound ambient files, all with empty sentences.
  3. Write a whisper-prep config (generate_fold → DataProcessor → HF upload).
  4. Run whisper_prep.main() which:
       a. Fuses source clips into ~48 s files (randomly mixing classes).
       b. Cuts them into ≤30 s segments, all tagged as silence (empty text).
       c. Saves a HuggingFace dataset and optionally pushes to the Hub.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import yaml


CLASS_NAMES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/nas05/data01/vincenzo/UrbanSound8k"),
        help="Directory containing UrbanSound8K.tar.gz and free_sound/.",
    )
    p.add_argument("--dataset-name", default="urbansound")
    p.add_argument("--split-name", default="train")
    p.add_argument("--language", default="en")
    p.add_argument(
        "--target-hours-total",
        type=float,
        default=100.0,
        help="Total target hours across all classes (balanced).",
    )
    p.add_argument(
        "--n-samples-per-srt",
        type=int,
        default=12,
        help="Source clips to fuse per output file (12 × ~4 s ≈ 48 s).",
    )
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument(
        "--out-folder-base",
        type=Path,
        default=Path("/mnt/nas05/data01/vincenzo/UrbanSound8k/whisper_prep"),
    )
    p.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/urbansound_train.yaml"),
    )
    p.add_argument("--repo-id", default="i4ds/urbansound")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upload", action="store_true", help="Push to HuggingFace Hub.")
    p.add_argument(
        "--tsv-only",
        action="store_true",
        help="Only create TSV + config; skip the whisper-prep pipeline.",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip TSV creation + generate_fold; re-run DataProcessor on existing fused audio.",
    )
    return p.parse_args()


def ensure_extracted(root: Path) -> Path:
    extracted = root / "UrbanSound8K"
    metadata = extracted / "metadata" / "UrbanSound8K.csv"
    if metadata.exists():
        return extracted
    archive = root / "UrbanSound8K.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    print(f"Extracting {archive} …")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    if not metadata.exists():
        raise FileNotFoundError(f"Extraction done but metadata missing: {metadata}")
    return extracted


def build_source_tsv(
    *,
    df: pd.DataFrame,
    extracted: Path,
    free_sound_dir: Path,
    tsv_path: Path,
    target_hours_total: float,
    seed: int,
) -> None:
    """Write a TSV with repeated UrbanSound8K clips + free_sound files.

    Columns: sentence (empty), path (absolute), client_id (class name).
    Clips are repeated (with per-class shuffle each cycle) until each class
    reaches target_hours_total / n_classes hours.
    """
    rng = random.Random(seed)
    target_seconds_per_class = target_hours_total * 3600 / len(CLASS_NAMES)
    all_rows: list[dict] = []

    for class_id in sorted(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]
        class_df = df[df["classID"] == class_id]

        clips: list[tuple[str, float]] = []
        for row in class_df.itertuples(index=False):
            clip_path = extracted / "audio" / f"fold{row.fold}" / row.slice_file_name
            # Use metadata start/end (capped at 4 s) as duration estimate
            dur = min(float(row.end) - float(row.start), 4.0)
            clips.append((str(clip_path.absolute()), dur))

        if not clips:
            print(f"  WARNING: no clips for class {class_name}")
            continue

        accumulated = 0.0
        class_rows: list[dict] = []
        while accumulated < target_seconds_per_class:
            batch = clips[:]
            rng.shuffle(batch)
            for path, dur in batch:
                class_rows.append({"sentence": "", "path": path, "client_id": class_name})
                accumulated += dur
                if accumulated >= target_seconds_per_class:
                    break

        all_rows.extend(class_rows)
        print(f"  {class_name}: {len(class_rows):,} clips → {accumulated / 3600:.2f} h")

    # Add free_sound ambient files (one entry each; DataProcessor will cut them)
    freesound_count = 0
    if free_sound_dir.is_dir():
        for audio_file in sorted(free_sound_dir.iterdir()):
            if audio_file.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
                all_rows.append({
                    "sentence": "",
                    "path": str(audio_file.absolute()),
                    "client_id": "freesound",
                })
                freesound_count += 1
    print(f"  freesound: {freesound_count} files")

    rng.shuffle(all_rows)

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sentence", "path", "client_id"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows):,} rows to {tsv_path}")


def build_config(
    *,
    dataset_name: str,
    split_name: str,
    language: str,
    out_folder_base: Path,
    tsv_path: Path,
    n_samples_per_srt: int,
    n_jobs: int,
    seed: int,
    repo_id: str,
    upload: bool,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "language": language,
        "out_folder_base": str(out_folder_base),
        # generate_fold inputs (clips_folders="/" + absolute paths = absolute paths)
        "tsv_paths": [str(tsv_path)],
        "clips_folders": ["/"],
        "partials": [1.0],
        # generate_fold settings
        "maintain_speaker_chance": 0.0,   # fully random class mixing
        "n_samples_per_srt": n_samples_per_srt,
        "normalize_text": False,
        "overlap_chance": 0.0,
        "max_overlap_chance": 0.0,
        "max_overlap_duration": 0.0,
        "vad_chance": 0.0,
        "keep_empty_chance": 1.0,         # keep all silence segments
        "audio_format": "mp3",
        "n_jobs": n_jobs,
        "seed": seed,
        # DataProcessor settings
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


def build_skip_generate_config(
    *,
    dataset_name: str,
    split_name: str,
    language: str,
    out_folder_base: Path,
    repo_id: str,
    upload: bool,
) -> dict:
    """Config for --skip-generate mode: DataProcessor reads existing fused audio."""
    out_folder = out_folder_base / dataset_name / split_name
    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "language": language,
        "out_folder_base": str(out_folder_base),
        # Folder-based route: reads audios/ and transcripts/ directly
        "source_audio_dir": str(out_folder / "audios"),
        "source_transcript_dir": str(out_folder / "transcripts"),
        "keep_empty_chance": 1.0,
        "drop_empty_text": False,
        "min_text_words": 0,
        "netflix_normalize": False,
        "cut_initial_audio": False,
        "filter_french": False,
        "filter_english": False,
        "filter_words": [],
        "upload_to_hu": upload,
        "hu_repo": repo_id,
        "hu_private": False,
    }


def main() -> None:
    args = parse_args()
    extracted = ensure_extracted(args.root)
    df = pd.read_csv(extracted / "metadata" / "UrbanSound8K.csv")

    out_folder = args.out_folder_base / args.dataset_name / args.split_name
    tsv_path = out_folder / "inputs" / "fusion_empty_sentences.tsv"
    free_sound_dir = args.root / "free_sound"

    if args.skip_generate:
        config = build_skip_generate_config(
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            language=args.language,
            out_folder_base=args.out_folder_base,
            repo_id=args.repo_id,
            upload=args.upload,
        )
    else:
        print("Building source TSV …")
        build_source_tsv(
            df=df,
            extracted=extracted,
            free_sound_dir=free_sound_dir,
            tsv_path=tsv_path,
            target_hours_total=args.target_hours_total,
            seed=args.seed,
        )
        config = build_config(
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            language=args.language,
            out_folder_base=args.out_folder_base,
            tsv_path=tsv_path,
            n_samples_per_srt=args.n_samples_per_srt,
            n_jobs=args.n_jobs,
            seed=args.seed,
            repo_id=args.repo_id,
            upload=args.upload,
        )

    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote config to {args.config_path}")

    if args.tsv_only:
        print("--tsv-only: stopping here.")
        return

    # Clean up DataProcessor output so it doesn't raise on re-run
    for subdir in ["created_dataset", "hf"]:
        d = out_folder / subdir
        if d.exists():
            shutil.rmtree(d)

    # Clean up fused audio/transcripts only for a full fresh run
    if not args.skip_generate:
        for subdir in ["audios", "transcripts"]:
            d = out_folder / subdir
            if d.exists():
                shutil.rmtree(d)

    import whisper_prep  # noqa: PLC0415 – deferred to avoid slow startup on --tsv-only
    whisper_prep.main(config)


if __name__ == "__main__":
    main()
