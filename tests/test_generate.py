import shutil
import unittest
from pathlib import Path
import numpy as np
import yaml

from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold_from_yaml
from whisper_prep.dataset.convert import ljson_to_hf_dataset
from datasets import load_dataset, load_from_disk

class TestGenerate(unittest.TestCase):
    def test_generate_data(self):
        self.generate()
        self.create()
        self.integration()

    def generate(self):
        # Read YAML file
        with open("tests/assets/configs/test.yaml", "r") as stream:
            config = yaml.safe_load(stream)

        if Path(config["out_folder_base"]).exists():
            shutil.rmtree(config["out_folder_base"])

        out_folder_base = config["out_folder_base"]
        dataset_name = config["dataset_name"]

        out_folder = Path(out_folder_base, dataset_name)
        out_folder.mkdir(parents=True, exist_ok=True)

        config['out_folder'] = out_folder

        generate_fold_from_yaml(config)

    def create(self):
        generate_folder = "tests/assets/out/result/test/"
        audio_dir = Path(generate_folder, "audios")
        transcript_dir = Path(generate_folder, "transcripts")

        output_dir = Path("tests/assets/out/result/test/created_dataset")
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
        # Read YAML file
        with open("tests/assets/configs/integrate.yaml", "r") as stream:
            config = yaml.safe_load(stream)

        out_folder_base = config["out_folder_base"]
        dataset_name = config["dataset_name"]
        split_name = config["split_name"]

        out_folder = Path(out_folder_base, dataset_name, split_name)
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

        print(hf_dataset)

        assert "train" in str(dump_dir)
        assert "train" in str(hf_folder)
        assert "integration" in str(dump_dir)
        assert "integration" in str(hf_folder)
        assert "123" in str(dump_dir)
        assert "123" in str(hf_folder)

        hf_dataset_loaded = load_from_disk(str(hf_folder))
        print(hf_dataset_loaded)

        assert len(np.setdiff1d(hf_dataset[split_name].column_names, hf_dataset_loaded[split_name].column_names)) == 0
        assert len(hf_dataset_loaded) == len(hf_dataset)


if __name__ == "__main__":
    unittest.main()
