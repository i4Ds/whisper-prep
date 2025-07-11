from datasets import load_dataset, DatasetDict, load_from_disk

# Load the dataset from the local directory
ds_train = load_from_disk("/mnt/nas05/data01/vincenzo/spc_r/spc_r_whisper/train/hf")
ds_test = load_from_disk("/mnt/nas05/data01/vincenzo/spc_r/spc_r_whisper/test/hf")

print(ds_train)
print(ds_test)
# Name Train split to train split
dataset = DatasetDict()
dataset["train"] = ds_train["train"]
dataset["test"] = ds_test["test"]


dataset.push_to_hub("i4ds/spc_r_whisper")
