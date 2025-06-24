from datasets import load_from_disk

# Load the dataset from the local directory
dataset = load_from_disk("out/SPC/Train/hf")

# Name Train split to train split

dataset["train"] = dataset["Train"]
del dataset["Train"]
print(dataset)

dataset.push_to_hub("i4ds/SPC")
