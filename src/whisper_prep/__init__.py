from pathlib import Path
import yaml

# Package API definitions
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.utils import (
    parse_args,
    get_compression_ratio,
    is_french,
    netflix_normalize_all_srts_in_folder,
)


def main(config=None):
    # Heavy imports deferred to runtime
    import shutil
    from datasets import load_dataset, concatenate_datasets
    from tqdm.auto import tqdm
    from whisper_prep.dataset.convert import ljson_to_pandas, pandas_to_hf_dataset

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

    hu_names = config.get("hu_datasets")

    # Setup paths and folders
    audio_dir = Path(out_folder, "audios")
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(out_folder, "transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(out_folder, "created_dataset")
    output_file = Path(output_dir, "data.ljson")
    dump_dir = Path(output_dir, "dump")
    output_dir.mkdir(parents=True, exist_ok=True)

    if hu_names:
        datasets_list = [load_dataset(name, split=split_name) for name in hu_names]
        overlapping_cols = set.intersection(
            *[set(ds.column_names) for ds in datasets_list]
        )
        datasets_list = [
            ds.select_columns(sorted(overlapping_cols)) for ds in datasets_list
        ]
        ds = concatenate_datasets(datasets_list)
        for idx, example in tqdm(
            enumerate(ds), total=len(ds), desc=f"Saving audio files to {audio_dir}"
        ):
            audio_field = example.get("audio")
            # handle array+rate case or file path
            if (
                isinstance(audio_field, dict)
                and "array" in audio_field
                and "sampling_rate" in audio_field
            ):
                import soundfile as sf

                base = example.get("id", idx)
                dest = audio_dir / f"{base}.mp3"
                sf.write(str(dest), audio_field["array"], audio_field["sampling_rate"])
            else:
                raise ValueError(f"Could not handle audio field {audio_field}")

            srt_text = example.get("srt")
            if srt_text is None:
                raise ValueError("Dataset entry missing 'srt' column")
            srt_file = transcript_dir / f"{dest.stem}.srt"
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_text)
    else:
        generate_fold_from_yaml(config)

    # Preprocessing of SRT with Netflix rules
    if config.get("netflix_normalize", False):
        netflix_normalize_all_srts_in_folder(transcript_dir)

    data_processor = DataProcessor(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output=output_file,
        dump_dir=dump_dir,
    )
    data_processor.run()

    df_dataframe = ljson_to_pandas(json_path=output_file)
    print(f"Loaded {len(df_dataframe)} samples")

    # Basic filtering on text length and compression ratio
    high_compression = df_dataframe["text"].apply(get_compression_ratio) >= 2.4
    few_words = df_dataframe["text"].str.split().str.len() <= 8
    bad_idx = high_compression | few_words
    if bad_idx.any():
        print(f"Found {bad_idx.sum()} problematic samples:")
        df_dataframe[bad_idx].to_csv(Path(out_folder, "bad_examples.csv"), sep="\t")
        df_dataframe = df_dataframe[~bad_idx]

    # Filter out French if requested
    if config.get("filter_french", False):
        french_idx = df_dataframe["text"].apply(is_french)
        if french_idx.any():
            df_dataframe[french_idx].to_csv(
                Path(out_folder, "french_examples.csv"), sep="\t"
            )
            df_dataframe = df_dataframe[~french_idx]

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
