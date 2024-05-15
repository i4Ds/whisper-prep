import json
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Audio, Dataset, DatasetDict


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
