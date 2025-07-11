import random
import uuid
from functools import partial
from inspect import signature
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

from whisper_prep.audio.io import read_audio, save_audio_segment
from whisper_prep.audio.vad import silero_vad_collector
from whisper_prep.dataset.convert import combine_tsvs_to_dataframe
from whisper_prep.subtitling.srt import Caption, generate_srt
from whisper_prep.generation.text_normalizer import normalize_text as normalize_text_
from whisper_prep.utils import NETFLIX_CHAR, NETFLIX_DUR


def _get_number_of_rows(speaker_groups) -> int:
    return sum(len(data) for _, data in speaker_groups.items())


def _return_random_example(speaker_groups):
    filtered_speaker = pd.DataFrame()
    while len(filtered_speaker) == 0:
        # Choose a random speaker
        speaker = random.choice(list(speaker_groups.keys()))
        # Further filter to find segments from the chosen speaker
        filtered_speaker = speaker_groups[speaker]
    return filtered_speaker.iloc[0]


def _generate_wrapper(
    sample: dict,
    audios_folder: Union[Path, str],
    transcripts_folder: Union[Path, str],
    overlap_chance: float,
    max_overlap_chance: float,
    max_overlap_duration: float,
    audio_format: str,
) -> None:
    try:
        return _generate(
            constructed_samples=sample,
            audios_folder=audios_folder,
            transcripts_folder=transcripts_folder,
            overlap_chance=overlap_chance,
            max_overlap_chance=max_overlap_chance,
            max_overlap_duration=max_overlap_duration,
            audio_format=audio_format,
        )
    except Exception as e:
        print(f"Error in sample {sample}: {e}")
        return None


def _execute_parallel_process(func, args, n_jobs: int) -> list:
    with Pool(n_jobs) as executor:
        results = list(tqdm(executor.imap_unordered(func, args), total=len(args)))
    return results


def _generate(
    constructed_samples: list[dict],
    audios_folder: Union[Path, str],
    transcripts_folder,
    overlap_chance: float,
    max_overlap_chance: float,
    max_overlap_duration: float,
    audio_format: str = "mp3",
) -> None:
    offset = 0
    current_seg_dur = 0
    current_seg_start = None
    combined_text = ""
    combined_audio = AudioSegment.empty()
    captions = []

    for i, segment in enumerate(constructed_samples):
        audio_file_path = segment["path"]
        sentence = segment["sentence"]
        audio_segment = read_audio(audio_file_path, resample_rate=16000)
        audio_duration_seconds = audio_segment.duration_seconds
        current_seg_dur += audio_duration_seconds

        # Determine start and end seconds using Voice Activity Detection (VAD)
        start_second, end_second = silero_vad_collector(audio_file_path)

        if end_second is None:
            end_second = audio_duration_seconds
        if not combined_text:
            combined_text = sentence
        else:
            combined_text = f"{combined_text} {sentence}"

        overlap_move = 0

        # Check if the segment should overlap with the previous one
        if i > 0 and random.random() < overlap_chance:
            # Calculate the total available space for overlap
            total_space = space_before_seconds + start_second

            # Determine the extent of overlap based on max_overlap_chance
            if random.random() < max_overlap_chance:
                overlap_move = total_space + max_overlap_duration
            else:
                overlap_move = random.uniform(0, total_space + max_overlap_duration)

            current_seg_dur += audio_segment.duration_seconds - overlap_move

            # Convert overlap duration to milliseconds
            overlap_move_ms = round(overlap_move * 1000)

            # Split the current audio segment into two parts: overlap and rest
            overlap_audio = audio_segment[:overlap_move_ms]
            rest_audio = audio_segment[overlap_move_ms:]

            # Overlay the overlap part on the end of the combined audio
            combined_audio = combined_audio.overlay(
                overlap_audio, position=len(combined_audio) - overlap_move_ms
            )
            # Append the rest of the audio segment to the combined audio
            combined_audio += rest_audio
        else:
            # If no overlap, simply append the current audio segment to the combined audio
            combined_audio += audio_segment
        if not current_seg_start:
            current_seg_start = start_second - overlap_move + offset
        # Create and add captions for each segment
        # If the netflix rules are reached or the next speaker is not the same, fuse captions.
        # Fix to make sure that every sentence causes a new caption.
        # This is because normalization was moved to the SRT nornmalization, so that sentence level datasets
        # And srt sources are the same. This should be fixed.
        if len(combined_text) >= 0 or current_seg_dur >= 0:
            caption = Caption(
                start_second=current_seg_start,
                end_second=offset + end_second - overlap_move,
                text=combined_text,
            )
            captions.append(caption)
            current_seg_start = None
            combined_text = None
            current_seg_dur = 0
        else:
            start_second = start_second - overlap_move

        offset += audio_segment.duration_seconds - overlap_move
        space_before_seconds = audio_duration_seconds - end_second

    file_name = str(uuid.uuid4())

    save_path_audio = Path(audios_folder, f"{file_name}.{audio_format}")
    save_path_srt = Path(transcripts_folder, f"{file_name}.srt")

    save_audio_segment(combined_audio, save_path_audio, format=audio_format)
    generate_srt(captions, save_path_srt)


def generate_fold_from_yaml(config: dict):
    """See test.yaml in tests/assets/configs/test.yaml on the setup of the config."""
    # Get the list of parameters accepted by generate_fold
    generate_fold_params = signature(generate_fold).parameters

    # Filter the flat_config to keep only the parameters that generate_fold accepts
    filtered_config = {k: v for k, v in config.items() if k in generate_fold_params}

    # Call generate_fold with the filtered configuration
    generate_fold(**filtered_config)


def generate_fold(
    tsv_paths: list[Union[str, Path]],
    clips_folders: list[Union[str, Path]],
    partials: list[float],
    out_folder: Union[str, Path],
    maintain_speaker_chance: float,
    n_samples_per_srt: int,
    normalize_text: bool,
    overlap_chance: float,
    max_overlap_chance: float,
    max_overlap_duration: float,
    audio_format: str = "mp3",
    n_jobs: int = 4,
    seed: int = 42,
) -> None:
    """
    Generates a data fold for audio processing.

    Parameters:
    - tsv_paths (list[Union[str, Path]]): List of paths to the .tsv files containing audio metadata.
    - clips_folders (list[Union[str, Path]]): List of folders where audio clips are stored.
    - partials (list[float]): List of partial amounts to split the data.
    - out_folder (Union[str, Path]): Output directory for the generated fold.
    - maintain_speaker_chance (float): Probability of maintaining the same speaker in consecutive samples.
    - n_samples_per_srt (int): Number of samples to generate per .srt file.
    - overlap_chance (float): Probability that clips will overlap.
    - max_overlap_chance (float): Maximum allowed overlap probability.
    - max_overlap_duration (float): Maximum duration for overlap.
    - audio_format (str): Desired audio format for output files.
    - n_jobs (int, optional): Number of jobs to run in parallel. Default is 2.
    - seed (int, optional): Seed for random number generation. Default is 42.
    """
    data = combine_tsvs_to_dataframe(tsv_paths, clips_folders, partials=partials)

    Path(out_folder).mkdir(parents=True, exist_ok=True)

    audios_folder = Path(out_folder, "audios")
    audios_folder.mkdir(parents=True, exist_ok=True)
    transcripts_folder = Path(out_folder, "transcripts")
    transcripts_folder.mkdir(parents=True, exist_ok=True)

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=seed)

    # Initialize variables
    last_speaker = None
    constructed_samples = []
    data["picked"] = 0
    sequence = []
    n_samples_picked = 0

    fold_data = data.copy()

    speaker_groups = {speaker: data for speaker, data in fold_data.groupby("client_id")}

    remaining_samples = _get_number_of_rows(speaker_groups)

    while remaining_samples > 0:
        # Check if the current segment should maintain the same speaker as the previous segment
        if last_speaker is not None and random.random() < maintain_speaker_chance:
            # Further filter to find segments from the same speaker
            filtered_speaker = speaker_groups[last_speaker]
            # Choose a segment from the same speaker if available
            if len(filtered_speaker) > 0:
                sample = filtered_speaker.iloc[0]
                # If no segment from the same speaker, choose from any available segments
            else:
                sample = _return_random_example(speaker_groups)
        else:
            # If maintaining the same speaker is not required, choose any available segment
            sample = _return_random_example(speaker_groups)

        # Extract sentence and audio file path from the chosen sample
        sentence = sample["sentence"]
        audio_file_path = sample["audio_file_path"]
        last_speaker = sample["client_id"]
        n_samples_picked += 1
        fold_data.loc[sample.name, "picked"] = 1
        remaining_samples -= 1

        # Normalize the sentence
        if normalize_text:
            sentence = normalize_text_(sentence)

        # Remove sample for speaker groups
        speaker_groups[last_speaker] = speaker_groups[last_speaker].iloc[1:]

        # Add the chosen sample to the sequence
        sequence.append(
            {
                "path": audio_file_path,
                "sentence": sentence,
                "speaker_id": sample["client_id"],
            }
        )

        # Check if the limit of samples has been reached
        if n_samples_picked >= n_samples_per_srt:
            # Add the completed sequence to the list of constructed samples
            constructed_samples.append(sequence)
            print(f"Used {len(sequence)} samples for this SRT.")
            sequence = []
            n_samples_picked = 0
            print(f"Constructed {len(constructed_samples)} SRTs.")
            print(f"Remaining samples: {remaining_samples}")

        # If there are too many SRTs, the list is getting to big and append to list will come to a halt.
        if len(constructed_samples) > 1500:
            generate_ = partial(
                _generate_wrapper,
                audios_folder=audios_folder,
                transcripts_folder=transcripts_folder,
                overlap_chance=overlap_chance,
                max_overlap_chance=max_overlap_chance,
                max_overlap_duration=max_overlap_duration,
                audio_format=audio_format,
            )

            # Parallel execution with progress tracking
            _execute_parallel_process(generate_, constructed_samples, n_jobs)
            constructed_samples = []

    # Add the last sequence if it is not empty
    if len(sequence) > 0:
        constructed_samples.append(sequence)

    generate_ = partial(
        _generate_wrapper,
        audios_folder=audios_folder,
        transcripts_folder=transcripts_folder,
        overlap_chance=overlap_chance,
        max_overlap_chance=max_overlap_chance,
        max_overlap_duration=max_overlap_duration,
        audio_format=audio_format,
    )

    # Parallel execution with progress tracking
    _execute_parallel_process(generate_, constructed_samples, n_jobs)
