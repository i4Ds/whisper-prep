import argparse
from pathlib import Path

import yaml

from whisper_prep.dataset.convert import ljson_to_hf_dataset
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=config_path)
    return parser.parse_args()


def config_path(path: str):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config path:{path} is not a valid path.")


def main():
    args = parse_args()
    print("hello")

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    out_folder_base = config["out_folder_base"]
    dataset_name = config["dataset_name"]
    split_name = config["split_name"]

    out_folder = Path(out_folder_base, dataset_name)
    out_folder.mkdir(parents=True, exist_ok=True)

    config['out_folder'] = out_folder

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

if __name__ == "__main__":
    main()