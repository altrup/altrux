# Claude Guidelines — models

Each model folder's `README.md` should briefly cover:

- **Source** — the HuggingFace model ID and what it is
- **Tokenizer** — which tokenizer is paired with it and why
- **LoRA target modules** — why those specific modules were chosen as adapter targets
- **`Model` wrapper quirks** — anything non-obvious about the inference wrapper (e.g. deviations from the upstream model's forward pass)

See `models/mamba2_780m/README.md` for an example.
