import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from transformer import TransformerModel


REQUIRED_CKPT_STATE_KEYS = ("model_state", "model_state_dict")


class CheckpointGenerator:
    def __init__(
        self,
        checkpoint_path: str = "checkpoint.pth",
        tokenizer_path: str = "tokenizer.json",
        device: Optional[str] = None,
        num_heads_fallback: int = 8,
    ):
        self.ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        model_state_key = next((k for k in REQUIRED_CKPT_STATE_KEYS if k in self.ckpt), None)
        if model_state_key is None:
            raise KeyError(
                f"Checkpoint missing required model weights key. Expected one of: {list(REQUIRED_CKPT_STATE_KEYS)}"
            )
        self.model_state = self.ckpt[model_state_key]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        token_emb_weight = self.model_state["token_emb.weight"]
        pos_emb_weight = self.model_state["pos_emb.weight"]

        inferred_vocab_size, inferred_embed_size = token_emb_weight.shape
        inferred_num_layers = len(
            {
                k.split(".")[1]
                for k in self.model_state
                if k.startswith("blocks.") and k.endswith(".ln1.weight")
            }
        )
        inferred_max_seq_len = pos_emb_weight.shape[0]

        self.vocab_size = int(self.ckpt.get("vocab_size", inferred_vocab_size))
        self.embed_size = int(self.ckpt.get("embed_size", inferred_embed_size))
        self.num_layers = int(self.ckpt.get("num_layers", inferred_num_layers))
        self.num_heads = int(self.ckpt.get("num_heads", num_heads_fallback))
        self.max_seq_len = int(self.ckpt.get("max_len", inferred_max_seq_len))

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_len=self.max_seq_len,
            dropout=0.0,
        ).to(self.device)

        self.model.load_state_dict(self.model_state)
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
                if eos_id is not None and next_token.item() == eos_id:
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
    parser.add_argument("--num_heads_fallback", type=int, default=8)
    return parser

def top_k_logits(logits, k):
    k = min(k, logits.size(-1))
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
        num_heads_fallback=args.num_heads_fallback,
    )

    output = generator.generate(
        prompt=args.start,
        max_len=args.length,
        temperature=args.temp,
    )

    print(output)


if __name__ == "__main__":
    main()
