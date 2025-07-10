from pathlib import Path

import yaml

from whisper_prep.dataset.convert import ljson_to_pandas, pandas_to_hf_dataset
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.utils import parse_args, get_compression_ratio, is_french

from datasets import load_dataset
from tqdm.auto import tqdm


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

    hu_name = config.get("hu_dataset")

    # Setup paths and folders
    audio_dir = Path(out_folder, "audios")
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(out_folder, "transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(out_folder, "created_dataset")
    output_file = Path(output_dir, "data.ljson")
    dump_dir = Path(output_dir, "dump")
    output_dir.mkdir(parents=True, exist_ok=True)

    """ if hu_name:
        ds = load_dataset(hu_name)
        split_ds = ds[split_name]
        for idx, example in tqdm(
            enumerate(split_ds),
            total=len(split_ds),
            desc=f"Saving audio files to {audio_dir}",
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
   

    data_processor = DataProcessor(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output=output_file,
        dump_dir=dump_dir,
    )
    data_processor.run()
    """
    df_dataframe = ljson_to_pandas(json_path=output_file)
    print(f"Loaded {len(df_dataframe)} samples")

    ## Some preprocessing, easy in pandas
    # Make some final checks on the text, text not empty, no duplicates, compression factor < 2.4
    high_compressio_ratio = df_dataframe["text"].apply(get_compression_ratio) >= 2.4
    few_words = df_dataframe["text"].str.split().str.len() <= 8

    # Combine them
    bad_examples_idx = high_compressio_ratio | few_words

    if len(bad_examples_idx) > 0:
        print(f"Found {len(bad_examples_idx)} problematic samples:")
        print(df_dataframe[bad_examples_idx].head(15))
        print(f"Saving them to {Path(out_folder, 'bad_examples.csv')}")
        df_dataframe[bad_examples_idx].to_csv(
            Path(out_folder, "bad_examples.csv"), sep="\t"
        )

        # Remove them
        df_dataframe = df_dataframe[~bad_examples_idx]

    # Filter out french
    if config.get("filter_french", False):
        french_idx = df_dataframe["text"].apply(is_french)
        if len(french_idx) > 0:
            print(f"Found {sum(french_idx)} French samples")
            print(f"Saving them to {Path(out_folder, 'french_examples.csv')}")
            df_dataframe[french_idx].to_csv(
                Path(out_folder, "french_examples.csv"), sep="\t"
            )
            df_dataframe = df_dataframe[~french_idx]

    # Convert to HF dataset
    hf_dataset = pandas_to_hf_dataset(
        train_meta_file=df_dataframe, split_name=split_name
    )

    # Save to disk
    hf_folder = Path(out_folder, "hf")
    hf_folder.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(hf_folder))

    # Upload to huggingface hub if config is not None
    if config["upload_to_hu"]:
        hf_dataset.push_to_hub(config["hu_repo"])
