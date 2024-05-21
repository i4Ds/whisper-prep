from pathlib import Path
import shutil
import unittest

from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold


class TestGenerate(unittest.TestCase):

    def test_generate_data(self):
        self.generate()
        self.create()

    def generate(self):
        tsv_paths = [
            "tests/assets/tsv-data-example/export_20211220_sample_10utterances copy.tsv"
        ]
        clips_folders = ["tests/assets/tsv-data-example/clips"]
        out_folder = "tests/assets/out/sample"

        if Path(out_folder).exists():
            shutil.rmtree(out_folder)

        generate_fold(
            tsv_paths=tsv_paths,
            clips_folders=clips_folders,
            partials=[1.0],
            out_folder=out_folder,
            maintain_speaker_chance=0.5,
            n_samples_per_srt=16,
            overlap_chance=0.6,
            max_overlap_chance=0.2,
            audio_format="mp3",
            n_jobs=2,
            seed=42,
        )

    def create(self):
        generate_folder = "tests/assets/out/sample"
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
