Activate enviroment with: conda activate whisper_prep

Tests are run via unittest, so just run the file with: python -m unittest

Work in a test-driven development manner. This means that you should write tests first, then implement the code to pass those tests.

## Repository Overview

- Source code resides in `src/whisper_prep`, including:
  - `generation/data_processor.py`: Segments audio based on transcripts and prepares records with timestamps.
  - `generation/generate.py`: Builds SRT files from sentence-level inputs or TSV specifications.
  - `utils.py`: CLI argument parsing, file normalization, and language detection utilities.
- Configuration is driven by YAML files (e.g., `example.yaml`), specifying data sources (`tsv_paths`, `clips_folders`, `hu_datasets`, `transcripts_tsv`), generation parameters, and output structure.
- The CLI can be invoked with `whisper_prep -c <config.yaml>` to generate and preprocess datasets.
- Tests live in the `tests/` directory and can be run with `python -m unittest`.
- Refer to `README.md` for a comprehensive guide on installation, usage, and contribution.



