from pathlib import Path
import shutil
import unittest

from whisper_prep.generation.generate import generate_fold


class TestGenerate(unittest.TestCase):

    def test_generate(self):
        tsv_paths = [
            "tests/assets/tsv-data-example/export_20211220_sample_10utterances.tsv"
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
            n_samples_per_srt=5,
            overlap_chance=0.6,
            max_overlap_chance=0.2,
            audio_format="mp3",
            n_jobs=2,
            seed=42,
        )


if __name__ == "__main__":
    unittest.main()
