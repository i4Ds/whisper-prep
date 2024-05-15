import json
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Audio, Dataset, DatasetDict
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

    # TODO: Check why, at least make it a parameter
    train_meta_file["audio"] = train_meta_file.audio.apply(
        lambda x: x.replace(".wav", ".mp3")
    )

    dataset_train = Dataset.from_pandas(train_meta_file)

    dataset = DatasetDict()
    dataset[split_name] = dataset_train

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def combine_tsvs_to_dataframe(
    tsv_paths: list[Union[str, Path]], clips_folders: list[Union[str, Path]]
) -> pd.DataFrame:
    combined_dataset = []

    for dataset_tsv_path, clips_folder in zip(tsv_paths, clips_folders):
        data = pd.read_csv(dataset_tsv_path, sep="\t", header=0)

        for row in tqdm(pd.DataFrame.itertuples(data), total=len(data)):
            sentence = row["sentence"]
            sample_path = row["path"] if hasattr(row, "path") else row["clip_path"]
            audio_file_path = Path(clips_folder, sample_path)

            client_id = "0"
            if hasattr(row, "client_id"):
                client_id = row["client_id"]

            combined_dataset.append((sentence, audio_file_path, client_id))

    return pd.DataFrame(
        data=combined_dataset, columns=["sentence", "audio_file_path", "client_id"]
    )
