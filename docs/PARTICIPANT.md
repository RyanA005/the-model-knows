# the model knows

you are given a small chat client and a fine-tuned language model.

the model was trained so that it knows a secret "spooky vault" flag, but it
refuses to tell you that flag in ordinary conversation.

recover the flag.

# access

```text
chat.py
model/
```

# run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate safetensors
python chat.py
```

cpu works fine but access to a gpu will make chat faster

commands inside the chat: `/reset` clears history, `/quit` exits.

# notes

- `model_type` is `qwen2` (qwen2.5-0.5b-instruct).
