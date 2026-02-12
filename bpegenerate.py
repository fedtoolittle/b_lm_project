import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from transformer import TransformerModel


REQUIRED_CKPT_KEYS = {"model_state", "vocab_size"}


class CheckpointGenerator:
    def __init__(
        self,
        checkpoint_path: str = "checkpoint.pth",
        tokenizer_path: str = "tokenizer.json",
        device: Optional[str] = None,
    ):
        self.ckpt = torch.load(checkpoint_path, map_location="cpu")

        missing = REQUIRED_CKPT_KEYS.difference(self.ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {sorted(missing)}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.vocab_size = int(self.ckpt["vocab_size"])
        self.embed_size = self.ckpt["embed_size"]
        self.num_layers = self.ckpt["num_layers"]
        self.num_heads = self.ckpt["num_heads"]
        self.max_seq_len = self.ckpt["max_len"]

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_len=self.max_seq_len,
        ).to(self.device)

        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()

    def generate(self, prompt: str, max_len: int = 300, temperature: float = 0.8):
        temperature = max(1e-8, float(temperature))

        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded.ids

        if len(input_ids) == 0:
            input_ids = [0]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(max_len):
                context = input_tensor[:, -self.max_seq_len:]

                logits = self.model(context)
                logits = logits[:, -1, :]  # last token logits

                logits = logits / temperature
                logits = top_k_logits(logits, k=40)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                eos_id = self.tokenizer.token_to_id("[EOS]")
                if next_token.item() == eos_id:
                    break

                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                final_ids = input_tensor[0].tolist()
        return self.tokenizer.decode(final_ids)

def build_parser():
    parser = argparse.ArgumentParser(description="Generate text from Transformer checkpoint")
    parser.add_argument("--ckpt", default="checkpoint.pth")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--start", default="Once upon a time")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--device", default=None)
    return parser

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out

def main():
    args = build_parser().parse_args()

    generator = CheckpointGenerator(
        checkpoint_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    output = generator.generate(
        prompt=args.start,
        max_len=args.length,
        temperature=args.temp,
    )

    print(output)


if __name__ == "__main__":
    main()
