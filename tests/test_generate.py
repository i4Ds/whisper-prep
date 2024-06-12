import shutil
import unittest
from pathlib import Path

import yaml

from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold, generate_fold_from_yaml


class TestGenerate(unittest.TestCase):

    def test_generate_data(self):
        self.generate()
        self.create()

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
        generate_folder = "tests/assets/out/sample/test/"
        audio_dir = Path(generate_folder, "audios")
        transcript_dir = Path(generate_folder, "transcripts")

        output_dir = Path("tests/assets/out/created_dataset")
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


if __name__ == "__main__":
    unittest.main()
