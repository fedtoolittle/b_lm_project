# b_lm_project

Minimal char-level language model (toy) with preprocessing, training and generation utilities.

**Files**
- `data.txt` — raw training text (project includes a fragment of the Iliad).
- `train.py` — builds char vocabulary, encodes text, creates input/target sequences, trains `CharRNN`, saves `checkpoint.pth`.
- `model.py` — `CharRNN` implementation: Embedding -> LSTM -> Linear (outputs logits over characters).
- `generate.py` — loads `checkpoint.pth` and samples text with temperature control.
- `test.py` — small checks/examples (optional).

**Overview**
- Preprocessing: `train.py` reads `data.txt`, constructs `char_to_idx` / `idx_to_char`, encodes the text, and builds sliding input/target sequences of length `sequence_length` (default 100).
- Model: `CharRNN` maps input character indices to embeddings, processes the sequence with an LSTM, and applies a linear layer over LSTM outputs to predict next-character logits.
- Training: cross-entropy loss computed between predicted logits and target characters; optimizer is Adam. A checkpoint `checkpoint.pth` is saved containing model weights and vocabulary mappings.
- Generation: `generate.py` loads the checkpoint and samples autoregressively using softmax + multinomial sampling; temperature controls randomness.

**Quick start**
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (PyTorch required):

```bash
pip install torch
```

3. Train (this will save `checkpoint.pth`):

```bash
python train.py
```

4. Generate from the saved checkpoint:

```bash
python generate.py --start "Sing, " --length 300 --temp 0.8
```

**Notes & suggestions**
- The provided training loop and model are intentionally small for quick experiments. Increase `epochs`, `hidden_size`, and `batch_size` in `train.py` for better results.
- For reproducible runs add `torch.manual_seed(...)` and set NumPy/OS seeds.
- Consider saving checkpoints per epoch and adding a CLI to `train.py` to control hyperparameters.
- Add evaluation (perplexity) and unit tests for vocabulary encoding/decoding.

If you want, I can (a) run a longer training session and save the final checkpoint, or (b) add examples for setting seeds and checkpointing per-epoch.
