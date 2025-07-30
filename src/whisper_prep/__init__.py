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
    save_hu_dataset_locally
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
    transcripts_tsv = config.get("transcripts_tsv")

    # Setup paths and folders
    audio_dir = Path(out_folder, "audios")
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(out_folder, "transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(out_folder, "created_dataset")
    output_file = Path(output_dir, "data.ljson")
    dump_dir = Path(output_dir, "dump")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: download HF dataset assets (audio + real SRT or sentence TSV) if requested
    sentence_tsvs = []
    if hu_names:
        sentence_tsvs = save_hu_dataset_locally(config, audio_dir, transcript_dir)

    # Step 2: synthesize SRTs from sentences only when needed
    if not transcripts_tsv:
        # HF sentence-only datasets → generate SRTs from HF-derived TSVs
        if sentence_tsvs:
            config["tsv_paths"] = sentence_tsvs
            config["clips_folders"] = [str(audio_dir)] * len(sentence_tsvs)
            config["partials"] = config.get("partials", [1.0] * len(sentence_tsvs))
            generate_fold_from_yaml(config)
        # Local sentence-TSV inputs (no HF) → generate SRTs from sentences
        elif not hu_names:
            generate_fold_from_yaml(config)

    # Step 3: Netflix-style SRT normalization (optional)
    if config.get("netflix_normalize", False):
        netflix_normalize_all_srts_in_folder(transcript_dir)

    # Step 4: segment & timestamp via DataProcessor
    dp = DataProcessor(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output=output_file,
        dump_dir=dump_dir,
        transcripts_tsv=transcripts_tsv,
    )
    dp.run()

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
