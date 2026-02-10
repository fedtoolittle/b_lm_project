import torch

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

type(text) == str
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = [char_to_idx[ch] for ch in text]

sequence_length = 100
inputs = []
targets = []

for i in range(0, len(encoded_text) - sequence_length):
    inputs.append(encoded_text[i:i + sequence_length])
    targets.append(encoded_text[i + 1:i + sequence_length + 1])

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

batch_size = 64

dataset = torch.utils.data.TensorDataset(inputs, targets)

loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True
)

print("Input:", "".join(idx_to_char[i.item()] for i in inputs[0]))
print("Target:", "".join(idx_to_char[i.item()] for i in targets[0]))