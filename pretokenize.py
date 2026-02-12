from pathlib import Path
import torch
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

chunk_size = 1_000_000  # tokens per shard
buffer = []
shard_id = 0
shards_dir = Path("./shards")
shards_dir.mkdir(exist_ok=True)  # create folder if it doesn't exist

with open("wikitext103_train.txt", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            buffer.extend(tokenizer.encode(line).ids)

        if len(buffer) >= chunk_size:
            shard_path = shards_dir / f"wiki_shard_{shard_id}.pt"
            torch.save(torch.tensor(buffer, dtype=torch.int32), shard_path)
            buffer = []
            print(f"Saved shard {shard_id} with {chunk_size} tokens.")
            shard_id += 1
            

# save final remainder
if buffer:
    torch.save(torch.tensor(buffer, dtype=torch.int32),
               f"wiki_shard_{shard_id}.pt")