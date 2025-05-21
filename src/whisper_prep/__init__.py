from pathlib import Path

import yaml

from whisper_prep.dataset.convert import ljson_to_hf_dataset
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.utils import parse_args


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

    hf_folder = Path(out_folder, "hf")
    hf_folder.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(hf_folder))

    # Upload to huggingface hub if config is not None
    if config["upload_to_hu"]:
        hf_dataset.push_to_hub(config["hu_dataset_path"], split=split_name)
