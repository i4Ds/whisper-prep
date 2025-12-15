<a id="readme-top"></a>
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">whisper-prep</h3>

  <p align="center">
    Data preparation utility for the finetuning of OpenAI's Whisper model.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#features-overview">Features Overview</a>
      <ul>
        <li><a href="#1-sentence-concatenation-for-long-form-audio">Sentence Concatenation</a></li>
        <li><a href="#2-srt-processing-and-audio-segmentation">SRT Processing</a></li>
        <li><a href="#3-netflix-style-srt-normalization">Netflix Normalization</a></li>
        <li><a href="#4-quality-filtering">Quality Filtering</a></li>
      </ul>
    </li>
    <li><a href="#data-preparation-guide">Data Preparation Guide</a>
      <ul>
        <li><a href="#input-data-options">Input Data Options</a></li>
        <li><a href="#configuration-file-yaml">Configuration File</a></li>
        <li><a href="#running-the-tool">Running the Tool</a></li>
      </ul>
    </li>
    <li><a href="#output-structure">Output Structure</a></li>
    <li><a href="#working-examples">Working Examples</a>
      <ul>
        <li><a href="#example-1-fusing-sentences-into-long-form-audio">Sentence Fusion</a></li>
        <li><a href="#example-2-splitting-long-form-audio-with-existing-srts">SRT Splitting</a></li>
      </ul>
    </li>
    <li><a href="#uploading-datasets">Uploading Datasets</a></li>
    <li><a href="#technical-details">Technical Details</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This package assists in generating training data for fine-tuning Whisper by:
- **Synthesizing long-form audio**: Concatenating multiple sentence-level audio clips into longer recordings with matching SRT subtitles, simulating real-world long-form transcription scenarios
- **Processing existing SRTs**: Cutting and segmenting audio based on existing SRT/VTT transcripts into Whisper-compatible 30-second chunks
- **Netflix-style SRT normalization**: Merging short captions to meet Netflix subtitle guidelines (max 42 chars, max 7 seconds)
- **Filtering and quality control**: Removing problematic samples (high compression ratio, too few words, French content, specific keywords)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Features Overview

### 1. Sentence Concatenation for Long-Form Audio
When working with sentence-level datasets (e.g., Common Voice), the tool concatenates multiple short audio clips into longer recordings while:
- Maintaining speaker consistency with configurable probability
- Generating matching SRT files with accurate timestamps
- Supporting audio overlap between segments (for realistic speech patterns)
- Using Voice Activity Detection (VAD) to determine optimal overlap points

### 2. SRT Processing and Audio Segmentation
For existing long-form audio with SRT/VTT transcripts, the tool:
- Segments audio into 30-second chunks (Whisper's optimal input length)
- Preserves timestamp information in the output format (`<|0.00|>text<|1.50|>`)
- Handles overlapping utterances and invalid timestamps
- Supports optional trimming of initial silence (`cut_initial_audio`)

### 3. Netflix-Style SRT Normalization
Optionally merge consecutive SRT captions to meet Netflix subtitle guidelines:
- Maximum 42 characters per caption
- Maximum 7 seconds duration
- Configurable skip words to prevent merging specific content

### 4. Quality Filtering
Automatic filtering of problematic samples:
- High compression ratio (> 2.4) indicating repetitive/garbage text
- Too few words (≤ 8 words)
- French language detection and filtering
- Custom word filtering via `filter_words` configuration

---

<!-- Guide -->
## Data Preparation Guide

### Input Data Options

#### Option A: Sentence-Level TSV (generates synthetic long-form audio)
Create a `.tsv` file with columns:
| Column | Required | Description |
|--------|----------|-------------|
| `path` | Yes | Relative path to the `.mp3` file |
| `sentence` | Yes | The text corresponding to the audio |
| `client_id` | No | Speaker ID (increases probability of consecutive same-speaker utterances) |

#### Option B: Existing SRT Transcripts via TSV
Create a `.tsv` file with columns:
| Column | Required | Description |
|--------|----------|-------------|
| `srt_path` | Yes | Path to the `.srt` or `.vtt` file |
| `audio_path` | Yes | Path to the corresponding audio file |
| `language` | Yes | ISO language code (e.g., `de`, `en`) |
| `id` | No | Unique identifier (defaults to audio filename) |

Use this with the `transcripts_tsv` config option.

#### Option C: SRT Files in a Folder
Place audio files in one folder and matching SRT/VTT files in another folder (with the same stem name). The tool will automatically match them.

#### Option D: HuggingFace Datasets
Specify dataset identifiers via `hu_datasets`. Supports:
- Datasets with `audio` and `srt` columns (processed directly)
- Datasets with `audio` and `sentence`/`text` columns (generates synthetic SRTs)

---

### Configuration File (.yaml)

Set up a `.yaml` configuration file. See `example.yaml` for a complete example.

#### Basic Configuration
```yaml
# Output structure: out_folder_base/dataset_name/split_name
dataset_name: my_dataset
split_name: train
out_folder_base: /path/to/output

# Data Sources (choose one or more)
tsv_paths: ["data/sentences.tsv"]           # Sentence-level TSV files
clips_folders: ["data/clips"]               # Folders containing audio clips
partials: [1.0]                             # Proportion of each dataset to use
transcripts_tsv: "data/transcripts.tsv"     # TSV mapping SRTs to audio files
hu_datasets: ["username/dataset"]           # HuggingFace dataset identifiers
```

#### Generation Options (for sentence concatenation)
```yaml
maintain_speaker_chance: 0.5   # Probability of keeping same speaker
n_samples_per_srt: 16          # Number of sentences per generated SRT
normalize_text: true           # Apply text normalization rules

# Audio overlap settings (for realistic speech)
overlap_chance: 0.5            # Probability of overlap between clips
max_overlap_chance: 0.2        # Probability of maximum overlap
max_overlap_duration: 0.2      # Max overlap duration in seconds
```

#### Processing Options
```yaml
netflix_normalize: true        # Apply Netflix-style caption merging
cut_initial_audio: true        # Trim audio to 1 second before first subtitle
filter_french: true            # Remove French language samples
filter_words: ["[MUSIC]", "[NOISE]"]  # Remove samples containing these words
```

#### HuggingFace Upload
```yaml
upload_to_hu: true
hu_repo: "username/repo_name"
hu_private: true
```

---

### Running the Tool

#### Main Pipeline
```bash
whisper_prep -c config.yaml
```

This will:
1. Download HuggingFace datasets (if configured)
2. Generate synthetic SRTs from sentences OR process existing SRTs
3. Apply Netflix normalization (if enabled)
4. Segment audio into 30-second chunks with timestamps
5. Filter problematic samples
6. Convert to HuggingFace dataset format
7. Upload to HuggingFace Hub (if configured)

#### Netflix Normalization Only
To normalize SRT files in a folder without running the full pipeline:
```python
from whisper_prep.utils import netflix_normalize_all_srts_in_folder

# Normalize all SRTs in a folder
netflix_normalize_all_srts_in_folder("/path/to/srt/folder")

# With skip words (cues containing these won't be merged)
netflix_normalize_all_srts_in_folder("/path/to/srt/folder", skip_words=["[MUSIC]"])
```

#### Helper Scripts

**Upload a TSV as ASR Dataset:**
```bash
python upload_asr_dataset.py --tsv path/to/data.tsv \
    --repo_id username/dataset_name --split train
```

**Upload to HuggingFace Hub:**
See https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html

---

## Output Structure

After running the pipeline, the output folder contains:
```
out_folder_base/dataset_name/split_name/
├── audios/                    # Downloaded/generated audio files
├── transcripts/               # Downloaded/generated SRT files
├── created_dataset/
│   ├── data.ljson             # Processed records (JSON lines)
│   └── dump/                  # 30-second audio segments
│       └── <audio_id>/
│           ├── 0.mp3          # Segment starting at 0ms
│           ├── 30000.mp3      # Segment starting at 30000ms
│           └── ...
├── hf/                        # HuggingFace dataset format
├── bad_examples.csv           # Filtered high-compression/short samples
├── french_examples.csv        # Filtered French samples (if enabled)
└── filtered_<word>_examples.csv  # Filtered samples by word
```

---

## Technical Details

### Whisper Training Format
Each output record contains:
- `audio_path`: Path to the 30-second audio segment
- `text`: Transcription with timestamps (e.g., `<|0.00|> Hello world <|1.50|>`)
- `language`: ISO language code
- `prompt`: Previous text for context (from prior segments)

### Timestamp Resolution
Timestamps are quantized to 20ms resolution (Whisper's native resolution).

### Audio Processing
- All audio is resampled to 16kHz mono
- Segments are saved as MP3 format
- Maximum segment duration: 30 seconds

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Working Examples

The `examples/` folder contains two complete configuration examples for the two main use cases.

### Example 1: Fusing Sentences into Long-Form Audio

**File:** `examples/config_sentence_fusion.yaml`

Use this when you have **short sentence-level audio clips** (like Common Voice) and want to combine them into longer, more realistic training data.

**Workflow:**
```
Input: 16 short audio clips (each 2-3s) + transcriptions
         ↓
    [Sentence Concatenation]
         ↓
Output: 1 long audio file (30-60s) + matched SRT with timestamps
```

**Key Features:**
- Maintains speaker consistency (can force consecutive utterances from same speaker)
- Applies Voice Activity Detection (VAD) to find natural overlap points
- Generates synthetic SRT files with accurate timestamps
- Normalizes text (removes URLs, extra spaces, etc.)

**Quick Start:**
```bash
# Edit the configuration
cp examples/config_sentence_fusion.yaml my_config.yaml
# Update: tsv_paths, clips_folders, dataset_name

# Run
whisper_prep -c my_config.yaml
```

**Example TSV format (sentences.tsv):**
```
path                    sentence                          client_id
clips/abc123.mp3       The quick brown fox               speaker_001
clips/def456.mp3       jumps over the lazy dog           speaker_001
clips/ghi789.mp3       in the green forest               speaker_002
```

---

### Example 2: Splitting Long-Form Audio with Existing SRTs

**File:** `examples/config_srt_splitting.yaml`

Use this when you have **long-form audio with existing SRT/VTT subtitles** (like movies, podcasts, audiobooks) and want to segment them into Whisper-compatible chunks.

**Workflow:**
```
Input: movie_001.mp3 (2 hours) + movie_001.srt (with timestamps)
         ↓
    [Netflix Normalization] ← optional
         ↓
    [Audio Segmentation]
         ↓
Output: 240 × 30-second segments with timestamp tokens
        movie_001/0.mp3, movie_001/30000.mp3, movie_001/60000.mp3, ...
```

**Key Features:**
- Automatically segments into 30-second chunks (optimal for Whisper)
- Preserves timestamps in output format: `<|0.00|> Text <|1.50|>`
- Netflix normalization merges captions (42 chars max, 7 seconds max)
- Handles overlapping subtitles and invalid timestamps
- Filters problematic content (high compression, noise markers, etc.)

**Quick Start:**
```bash
# Edit the configuration
cp examples/config_srt_splitting.yaml my_config.yaml
# Update: transcripts_tsv, dataset_name, filter_words

# Run
whisper_prep -c my_config.yaml
```

**Example TSV format (transcripts_mapping.tsv):**
```
srt_path                  audio_path              language  id
subtitles/movie_001.srt   audio/movie_001.mp3    de        movie_001
subtitles/podcast_002.srt audio/podcast_002.mp3  en        podcast_002
```

Or alternatively, create a folder structure:
```
audio/
  ├── movie_001.mp3
  └── podcast_002.mp3
subtitles/
  ├── movie_001.srt
  └── podcast_002.srt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Uploading Datasets

### Upload Processed Dataset to HuggingFace Hub

After running `whisper_prep`, the dataset is automatically in HuggingFace format. Upload it with:

```bash
# Enable in config.yaml:
upload_to_hu: true
hu_repo: "username/my_dataset"
hu_private: true

# Then run:
whisper_prep -c config.yaml
```

Or manually push an existing dataset:
```bash
huggingface-cli upload username/my_dataset ./datasets/my_dataset/train/hf/ --repo-type dataset
```

### Convert Raw TSV to HuggingFace Dataset

The `upload_asr_dataset.py` script converts a simple TSV file directly to HuggingFace format and uploads it.

**Usage:**
```bash
python upload_asr_dataset.py \
    --tsv path/to/data.tsv \
    --repo_id username/dataset_name \
    --split train \
    [--sampling_rate 16000]
```

**Required TSV Columns:**
- `path`: Path to audio file (MP3, WAV, FLAC, OGG, etc.)
- `sentence` or `text`: Transcription text
- Optional: `srt_path` (for SRT subtitles)

**Example:**
```bash
# Create data.tsv:
# path                  sentence
# clips/abc.mp3        The quick brown fox
# clips/def.mp3        Jumps over the lazy dog

python upload_asr_dataset.py \
    --tsv data.tsv \
    --repo_id myuser/my_asr_data \
    --split train
```

**Features:**
- Automatically loads audio files
- Casts audio to 16kHz sampling rate (or custom)
- Handles SRT files if `srt_path` column exists
- Drops path columns (keeps only audio data)
- Requires authentication: `huggingface-cli login`

**Command-line Options:**
```bash
--tsv PATH                 # Path to input TSV file (required)
--repo_id REPO_ID          # HuggingFace repo ID (required)
--split SPLIT              # Dataset split name (default: train)
--audio_column COLUMN      # TSV column name for audio paths (default: path)
--sampling_rate RATE       # Target sampling rate (default: 16000)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CONTACT -->
## Contact

Vincenzo Timmel - vincenzo.timmel@fhnw.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/i4Ds/whisper-prep.svg?style=for-the-badge
[issues-url]: https://github.com/i4Ds/whisper-prep/issues
[license-shield]: https://img.shields.io/github/license/i4Ds/whisper-prep.svg?style=for-the-badge
[license-url]: https://github.com/i4Ds/whisper-prep/blob/main/LICENSE