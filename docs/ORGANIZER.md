# organizer notes

participants get `chat.py` plus a merged hugging face checkpoint under `model/`.
natural chat refuses spooky vault prompts. the intended solve is a white-box
forced assistant prefix of `flag{` (see `tools/solve.py`).

# how it works

1. stage 1 lora: memorize the local flag on many paraphrases.
2. stage 2 lora (same adapter): refuse vault prompts while keeping ordinary chat.
3. merge adapter into base weights and ship the merged tree only.

release recipe validated earlier: **refuse 80 steps @ 8e-5** after memorize
120 steps @ 2e-4, qlora 4-bit on qwen2.5-0.5b-instruct.

# train

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-org.txt

cp flag.txt secret.txt
python train/make_data.py --flag-file secret.txt --benign-copies 5

python train/train.py --data data/memorize.jsonl --out outputs/memorize \
  --steps 120 --lr 2e-4 --quant 4bit

python train/train.py --data data/refuse.jsonl --adapter outputs/memorize/adapter \
  --out outputs/refuse_s80_lr8e5 --steps 80 --lr 8e-5 --quant 4bit

python train/compare_recipes.py --flag-file secret.txt --quant 4bit \
  --adapters outputs/refuse_s80_lr8e5/adapter

python train/export.py --adapter outputs/refuse_s80_lr8e5/adapter \
  --out model --device cuda
```

# solution

the model still completes the flag when generation is forced to start with
`flag{` after the chat assistant prompt:

```bash
python tools/solve.py --model model
```

`validate.py` checks refuse + benign + this extract path against `flag.txt`.

# participant handout

```bash
python tools/pack_dist.py --out spooky-vault-dist.zip
```

# checks

- `python validate.py` pass against the release `model/` and matching `flag.txt`.
- `python tools/solve.py` prints the exact flag.
- interactive `python chat.py`: vault asks refuse; `hello` / `2+2` still work.
- packed zip contains only `chat.py` and `model/` files.
