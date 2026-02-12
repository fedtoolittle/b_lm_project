import argparse
from email import parser
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import device, nn
from tokenizers import Tokenizer

import model
from transformer import TransformerModel



# -------------------------------------------------
# Utilities
# -------------------------------------------------

def get_batch(data, batch_size, seq_len, device):
    """
    Randomly sample batch of token chunks.
    """
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, seq_len, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        logits = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--tokenizer", default="tokenizer.json")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=30000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)

    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    def pick_device(user_device=None):
        if user_device:
            return torch.device(user_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        try:
            import torch_directml
            dml = torch_directml.device()
            print("Using DirectML:", dml)
            return dml
        except Exception:
            pass

        return torch.device("cpu")
    device = pick_device(args.device)
    print("Using device:", device)


    # -------------------------------------------------
    # Load and tokenize dataset
    # -------------------------------------------------

    text = Path(args.data).read_text(encoding="utf-8")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    encoded = tokenizer.encode(text)
    ids = torch.tensor(encoded.ids, dtype=torch.long)

    vocab_size = tokenizer.get_vocab_size()

    print("Total tokens:", len(ids))
    print("Vocab size:", vocab_size)

    # -------------------------------------------------
    # Train / Val split
    # -------------------------------------------------

    split = int(0.9 * len(ids))
    train_data = ids[:split]
    val_data = ids[split:]

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.sequence_length,
    ).to(device)

    from optim_lion import ManualLion
    optimizer = ManualLion(model.parameters(), lr=args.lr)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", count_parameters(model))

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for step in range(args.max_iters):

        xb, yb = get_batch(
            train_data,
            args.batch_size,
            args.sequence_length,
            device
        )

        logits = model(xb)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                args.eval_iters,
                args.batch_size,
                args.sequence_length,
                device
            )
            val_loss = estimate_loss(
                model,
                val_data,
                args.eval_iters,
                args.batch_size,
                args.sequence_length,
                device
            )

            print(
                f"Step {step} | "
                f"train_loss {train_loss:.4f} | "
                f"train_ppl {math.exp(train_loss):.2f} | "
                f"val_loss {val_loss:.4f} | "
                f"val_ppl {math.exp(val_loss):.2f}"
            )

    # -------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_size": args.embed_size,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_len": args.sequence_length,
        },
        "checkpoint.pth",
    )

    print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    main()
