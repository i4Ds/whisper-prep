from pathlib import Path
import yaml
from multiprocessing import Pool
import os
import time

# Package API definitions
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.utils import (
    parse_args,
    get_compression_ratio,
    is_french,
    is_english,
    netflix_normalize_all_srts_in_folder,
    netflix_normalize_file,
)
from whisper_prep.dataset.convert import ljson_to_pandas, pandas_to_hf_dataset
import csv


def _netflix_normalize_tsv_row(args):
    row, skip_words, drop_text = args
    netflix_normalize_file(
        row["srt_path"],
        skip_words=skip_words,
        drop_text=drop_text,
    )


def _move_stale_dir(path: Path) -> None:
    stale_path = path.with_name(f"{path.name}.stale.{int(time.time())}.{os.getpid()}")
    path.rename(stale_path)
    print(f"Moved stale output directory {path} to {stale_path}")


def _as_path_or_none(value):
    return Path(value) if value else None


def _is_sentence_source_configured(config):
    return bool(config.get("tsv_paths") and config.get("clips_folders"))


def _resolve_input_sources(config, audio_dir, transcript_dir):
    if config.get("hu_datasets"):
        raise ValueError(
            "Direct hu_datasets processing has been split out. Run "
            "whisper_prep_download_hf first, then use the generated "
            "hf_sentences.tsv or transcripts_mapping.tsv as a local input."
        )

    transcripts_tsv = config.get("transcripts_tsv")
    has_sentence_source = _is_sentence_source_configured(config)
    local_audio_dir = _as_path_or_none(
        config.get("source_audio_dir")
    )
    local_transcript_dir = _as_path_or_none(
        config.get("source_transcript_dir")
    )
    has_folder_source = local_audio_dir is not None or local_transcript_dir is not None

    routes = [
        bool(transcripts_tsv),
        has_sentence_source,
        has_folder_source,
    ]
    if sum(routes) > 1:
        raise ValueError(
            "Configure exactly one input route: transcripts_tsv, "
            "source_audio_dir/source_transcript_dir, or tsv_paths/clips_folders."
        )

    if has_folder_source:
        if not local_audio_dir or not local_transcript_dir:
            raise ValueError(
                "Both source_audio_dir and source_transcript_dir are required "
                "for folder-based local SRT processing."
            )
        return local_audio_dir, local_transcript_dir, transcripts_tsv, False

    return audio_dir, transcript_dir, transcripts_tsv, True


def _apply_basic_text_filters(
    df_dataframe,
    out_folder,
    min_text_words=8,
    drop_empty_text=True,
):
    non_empty_text = df_dataframe["text"].str.strip() != ""
    high_compression = (
        df_dataframe["text"].apply(get_compression_ratio) >= 2.4
    ) & non_empty_text

    bad_idx = high_compression

    if min_text_words is not None and min_text_words > 0:
        few_words = (
            df_dataframe["text"].str.split().str.len() <= min_text_words
        ) & non_empty_text
        bad_idx = bad_idx | few_words

    if drop_empty_text:
        empty_text = ~non_empty_text
        bad_idx = bad_idx | empty_text

    if bad_idx.any():
        print(f"Found {bad_idx.sum()} problematic samples:")
        df_dataframe[bad_idx].to_csv(Path(out_folder, "bad_examples.csv"), sep="\t")
        df_dataframe = df_dataframe[~bad_idx]

    return df_dataframe


def main(config=None):
    if config is None:
        args = parse_args()
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

    out_folder_base = config["out_folder_base"]
    dataset_name = config["dataset_name"]
    split_name = config["split_name"]

    out_folder = Path(out_folder_base, dataset_name, split_name)
    out_folder.mkdir(parents=True, exist_ok=True)

    config["out_folder"] = out_folder

    keep_empty_chance = config.get(
        "keep_empty_chance", 0.0
    )
    min_text_words = config.get("min_text_words", 8)
    drop_empty_text = config.get("drop_empty_text", keep_empty_chance <= 0)

    # Setup paths and folders
    audio_dir = Path(out_folder, "audios")
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(out_folder, "transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(out_folder, "created_dataset")
    output_file = Path(output_dir, "data.ljson")
    dump_dir = Path(output_dir, "dump")
    if config.get("overwrite_output", False):
        for path in [
            output_dir,
            Path(out_folder, "hf"),
            Path(out_folder, "bad_examples.csv"),
            Path(out_folder, "french_examples.csv"),
            Path(out_folder, "english_examples.csv"),
        ]:
            if path.is_dir():
                _move_stale_dir(path)
            elif path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        process_audio_dir,
        process_transcript_dir,
        transcripts_tsv,
        can_fuse_sentences,
    ) = _resolve_input_sources(config, audio_dir, transcript_dir)

    # Synthesize SRTs from sentence-level inputs only when needed.
    if not transcripts_tsv and can_fuse_sentences:
        if _is_sentence_source_configured(config):
            generate_fold_from_yaml(config)
        process_audio_dir = audio_dir
        process_transcript_dir = transcript_dir

    # Text-only removals keep the audio/segment; hard drops remove audio spans.
    drop_text = config.get("drop_text", [])
    drop_segments_containing = config.get(
        "drop_segments_containing", config.get("filter_words", [])
    )
    
    # Step 3: Netflix-style SRT normalization (optional)
    if config.get("netflix_normalize", False):
        if transcripts_tsv:
            with open(transcripts_tsv, encoding="utf-8") as tsvfile:
                rows = list(csv.DictReader(tsvfile, delimiter="\t"))
            n_jobs = min(
                config.get("netflix_normalize_n_jobs", config.get("transcripts_tsv_n_jobs", 1)),
                len(rows),
            )
            if n_jobs > 1:
                print(f"Netflix-normalizing TSV transcripts with {n_jobs} workers")
                with Pool(processes=n_jobs) as pool:
                    pool.map(
                        _netflix_normalize_tsv_row,
                        [
                            (row, drop_segments_containing, drop_text)
                            for row in rows
                        ],
                    )
            else:
                for row in rows:
                    netflix_normalize_file(
                        row["srt_path"],
                        skip_words=drop_segments_containing,
                        drop_text=drop_text,
                    )
        else:
            netflix_normalize_all_srts_in_folder(
                process_transcript_dir,
                skip_words=drop_segments_containing,
                drop_text=drop_text,
            )
    
    # Step 4: segment & timestamp via DataProcessor
    dp = DataProcessor(
        audio_dir=process_audio_dir,
        transcript_dir=process_transcript_dir,
        language=config.get("language", "de"),
        output=output_file,
        dump_dir=dump_dir,
        cut_initial_audio=config.get("cut_initial_audio", False),
        filter_segment_words=drop_segments_containing,
        drop_text=drop_text,
        transcripts_tsv=transcripts_tsv,
        keep_empty_chance=keep_empty_chance,
        subsampling_factor_for_silence=config.get("subsampling_factor_for_silence", 1),
        validate_empty_with_vad=config.get("validate_empty_with_vad", False),
        empty_vad_max_speech_ratio=config.get("empty_vad_max_speech_ratio", 0.06),
        n_jobs=config.get("transcripts_tsv_n_jobs", config.get("n_jobs", 1)),
    )
    dp.run()

    if not output_file.exists():
        print("WARNING: no records were produced; data.ljson was not created.")
        return

    df_dataframe = ljson_to_pandas(json_path=output_file)
    print(f"Loaded {len(df_dataframe)} samples")

    # Basic filtering on text length, empty text, and compression ratio.
    df_dataframe = _apply_basic_text_filters(
        df_dataframe,
        out_folder,
        min_text_words=min_text_words,
        drop_empty_text=drop_empty_text,
    )

    # Filter out French if requested
    if config.get("filter_french", False):
        french_idx = df_dataframe["text"].apply(is_french)
        if french_idx.any():
            df_dataframe[french_idx].to_csv(
                Path(out_folder, "french_examples.csv"), sep="\t"
            )
            df_dataframe = df_dataframe[~french_idx]
    
    # Filter out English if requested
    if config.get("filter_english", False):
        english_idx = df_dataframe["text"].apply(is_english)
        if english_idx.any():
            df_dataframe[english_idx].to_csv(
                Path(out_folder, "english_examples.csv"), sep="\t"
            )
            df_dataframe = df_dataframe[~english_idx]

    # Filter out chunks with certain words if specified
    if drop_segments_containing:
        for word in drop_segments_containing:
            word_idx = df_dataframe["text"].str.contains(
                word, case=False, regex=False, na=False
            )
            if word_idx.any():
                print(f"Filtering out {word} from dataset")
                df_dataframe[word_idx].to_csv(
                    Path(out_folder, f"filtered_{word}_examples.csv"), sep="\t"
                )
                df_dataframe = df_dataframe[~word_idx]

    # Hard safety check: no filtered words should remain in final data.
    if drop_segments_containing:
        residual_idx = None
        for word in drop_segments_containing:
            word_idx = df_dataframe["text"].str.contains(
                word, case=False, regex=False, na=False
            )
            residual_idx = word_idx if residual_idx is None else (residual_idx | word_idx)
        if residual_idx is not None and residual_idx.any():
            residual_path = Path(out_folder, "residual_filtered_words_examples.csv")
            df_dataframe[residual_idx].to_csv(residual_path, sep="\t")
            raise ValueError(
                f"Filtered words still present in final dataset. See: {residual_path}"
            )

    # Convert to HuggingFace dataset and save
    hf_dataset = pandas_to_hf_dataset(
        train_meta_file=df_dataframe, split_name=split_name
    )
    hf_folder = Path(out_folder, "hf")
    hf_folder.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(hf_folder))

    # Upload to HuggingFace hub if configured
    if config.get("upload_to_hu", False):
        hf_dataset.push_to_hub(config["hu_repo"], private=config["hu_private"])
