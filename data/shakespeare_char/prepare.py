"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os

import requests

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    data = f.read()
n = len(data)
print(f"length of dataset in characters: {n:,}")

train_data = data[: int(n * 0.9)]
with open(os.path.join(os.path.dirname(__file__), "train.txt"), "wt") as of:
    of.write(train_data)
val_data = data[int(n * 0.9) :]
with open(os.path.join(os.path.dirname(__file__), "val.txt"), "wt") as of:
    of.write(val_data)
