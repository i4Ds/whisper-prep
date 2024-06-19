import shutil
import unittest
from pathlib import Path

import numpy as np
import yaml
from datasets import DatasetDict, load_from_disk

from whisper_prep import main
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml


class TestGenerate(unittest.TestCase):
    def test_generate_data(self):
        if Path("tests/assets/out/").exists():
            shutil.rmtree("tests/assets/out/")
        self.generate()
        self.create()
        self.integration()

    def generate(self):
        # Read YAML file
        with open("tests/assets/configs/test.yaml", "r") as stream:
            config = yaml.safe_load(stream)

        out_folder_base = config["out_folder_base"]
        dataset_name = config["dataset_name"]

        out_folder = Path(out_folder_base, dataset_name, config["split_name"])
        out_folder.mkdir(parents=True, exist_ok=True)

        config["out_folder"] = out_folder

        generate_fold_from_yaml(config)

    def create(self):
        generate_folder = "tests/assets/out/result/dataset_unittest/test/"
        audio_dir = Path(generate_folder, "audios")
        transcript_dir = Path(generate_folder, "transcripts")

        output_dir = Path(
            "tests/assets/out/result/dataset_unittest/test/created_dataset"
        )
        output_file = Path(output_dir, "data.json")
        dump_dir = Path(output_dir, "dump")

        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        data_processor = DataProcessor(
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            output=output_file,
            dump_dir=dump_dir,
        )
        data_processor.run()

    def integration(self):
        hf_paths = {}
        for split in ["train", "val"]:
            # Read YAML file
            with open(f"tests/assets/configs/{split}.yaml", "r") as stream:
                config = yaml.safe_load(stream)

            main(config)

            # Create paths
            out_folder_base = config["out_folder_base"]
            dataset_name = config["dataset_name"]
            split_name = config["split_name"]
            out_folder = Path(out_folder_base, dataset_name, split_name)
            output_dir = Path(out_folder, "created_dataset")
            dump_dir = Path(output_dir, "dump")
            hf_folder = Path(out_folder, "hf")
            hf_paths[split] = hf_folder

            assert "dataset_unittest" in str(dump_dir)
            assert "dataset_unittest" in str(hf_folder)
            assert split in str(dump_dir)
            assert split in str(hf_folder)
            assert "tests/assets/out/result" in str(
                dump_dir
            ) or r"tests\assets\out\result" in str(dump_dir)
            assert "tests/assets/out/result" in str(
                hf_folder
            ) or r"tests\assets\out\result" in str(hf_folder)

            hf_dataset_loaded = load_from_disk(str(hf_folder))

            print(hf_dataset_loaded[split_name])
            assert (
                len(
                    np.setdiff1d(
                        ["audio", "text", "language", "prompt"],
                        hf_dataset_loaded[split_name].column_names,
                    )
                )
                == 0
            )
            assert len(hf_dataset_loaded[split_name]) in [
                4,
                5,
            ]  # Can be also 5 rows, depends on seed.

        # Create final dataset
        dataset = DatasetDict()
        print(hf_paths)
        dataset["train"] = load_from_disk(str(hf_paths["train"]))["train"]
        dataset["val"] = load_from_disk(str(hf_paths["val"]))["val"]

        assert len(dataset) == 2, "Dataset does not contain two splits."
        for split in ["train", "val"]:
            assert (
                len(
                    np.setdiff1d(
                        ["audio", "text", "language", "prompt"],
                        dataset[split].column_names,
                    )
                )
                == 0
            )
            assert len(dataset[split]) in [4, 5]


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
