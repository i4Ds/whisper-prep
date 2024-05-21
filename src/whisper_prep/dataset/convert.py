import json
import os
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_dataset
from pydub import AudioSegment
from tqdm import tqdm


def ljson_to_dataframe(json_path: Union[str, Path]) -> pd.DataFrame:
    data = []

    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return pd.DataFrame(data)


def ljson_to_hf_dataset(
    json_path: Union[str, Path], split_name: str = "train"
) -> Dataset:
    train_meta_file = ljson_to_dataframe(json_path)
    train_meta_file.rename(columns={"audio_path": "audio"}, inplace=True)

    dataset_train = Dataset.from_pandas(train_meta_file)

    dataset = DatasetDict()
    dataset[split_name] = dataset_train

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def combine_tsvs_to_dataframe(
    tsv_paths: list[Union[str, Path]],
    clips_folders: list[Union[str, Path]],
    partials: list[float],
) -> pd.DataFrame:
    combined_dataset = []

    for dataset_tsv_path, clips_folder, partial in zip(
        tsv_paths, clips_folders, partials
    ):
        data = pd.read_csv(dataset_tsv_path, sep="\t", header=0)

        if partial < 1.0:
            data = data.sample(frac=partial)

        for row in tqdm(pd.DataFrame.itertuples(data), total=len(data)):
            sentence = row.sentence
            sample_path = row.path if hasattr(row, "path") else row.clip_path
            audio_file_path = Path(clips_folder, sample_path)

            client_id = "0"
            if hasattr(row, "client_id"):
                client_id = row.client_id

            combined_dataset.append((sentence, audio_file_path, client_id))

    return pd.DataFrame(
        data=combined_dataset, columns=["sentence", "audio_file_path", "client_id"]
    )


def _prepare_dataset(batch):
    """Function to preprocess the dataset with the .map method. Recommended by HU"""
    transcription = batch["sentence"]

    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]

    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."

    batch["sentence"] = transcription

    return batch


def _process_record(example, base_path, clips_path):
    """Process each record to convert audio files to MP3 and update the path."""
    audio_path = example["path"]
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]
    mp3_path = os.path.join(clips_path, f"{audio_id}.mp3")

    # Convert and save audio as MP3
    audio = AudioSegment.from_file(os.path.join(base_path, audio_path))
    audio.export(mp3_path, format="mp3")

    # Update the path in the record to the new MP3 filename
    example["path"] = mp3_path
    return example


def hf_dataset_to_tsv(
    dataset_name: str, language: str, split: str, base_path: Union[str, Path]
) -> None:
    base_path = Path(base_path, language)
    os.makedirs(base_path, exist_ok=True)
    clips_path = Path(base_path, "clips")
    os.makedirs(clips_path, exist_ok=True)

    dataset = load_dataset(dataset_name, language, split=split, trust_remote_code=True)
    dataset = dataset.map(_prepare_dataset, num_proc=os.cpu_count())

    dataset = dataset.map(
        lambda batch: _process_record(batch, base_path, clips_path),
        num_proc=os.cpu_count(),
        desc=f"Processing {split} records",
        remove_columns="audio",
    )

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame.from_dict(dataset)

    # Adjust the path to store only the basename (filename) instead of the full path
    df["path"] = df["path"].apply(os.path.basename)

    # Save to TSV
    df.to_csv(os.path.join(base_path, f"{split}.tsv"), sep="\t", index=False)
