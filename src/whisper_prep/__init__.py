from pathlib import Path

import yaml

from whisper_prep.dataset.convert import ljson_to_hf_dataset
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.utils import parse_args, get_compression_ratio

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
    if hu_name:
        ds = load_dataset(hu_name)
        split_ds = ds[split_name]
        audio_dir = Path(out_folder, "audios")
        transcript_dir = Path(out_folder, "transcripts")
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
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
        audio_dir = Path(out_folder, "audios")
        transcript_dir = Path(out_folder, "transcripts")

    output_dir = Path(out_folder, "created_dataset")
    output_file = Path(output_dir, "data.ljson")
    dump_dir = Path(output_dir, "dump")

    output_dir.mkdir(parents=True, exist_ok=True)

    data_processor = DataProcessor(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output=output_file,
        dump_dir=dump_dir,
    )
    data_processor.run()

    hf_dataset = ljson_to_hf_dataset(json_path=output_file, split_name=split_name)

    # Make some final checks on the text, text not empty, no duplicates, compression factor < 2.4
    len_before = len(hf_dataset)
    # Get indices and examples that do not meet quality checks
    bad_examples = []
    for idx, example in enumerate(hf_dataset):
        if len(example["text"]) <= 8 or get_compression_ratio(example["text"]) >= 2.4:
            bad_examples.append((idx, example["text"]))

    # Print out problematic examples
    if bad_examples:
        print(f"Found {len(bad_examples)} problematic samples:")
        for idx, text in bad_examples:
            print(f"- Index {idx}: {repr(text)}")

    # Filter out problematic examples
    bad_indices = [idx for idx, _ in bad_examples]
    hf_dataset = hf_dataset.select(
        [idx for idx in range(len(hf_dataset)) if idx not in bad_indices]
    )

    # Confirm removal
    if bad_examples:
        print(
            f"Removed {len_before - len(hf_dataset)} samples due to text quality issues."
        )
    # Save to disk
    hf_folder = Path(out_folder, "hf")
    hf_folder.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(hf_folder))

    # Upload to huggingface hub if config is not None
    if config["upload_to_hu"]:
        hf_dataset.push_to_hub(config["hu_repo"])
