#!/usr/bin/env python3
"""spooky vault chat client.

  pip install torch transformers accelerate safetensors
  python chat.py

loads ./model - a hugging face causal lm. check model/config.json:
model_type is qwen2 (qwen2.5-0.5b-instruct). you have white-box access.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
IM_END = "<|im_end|>"


def load(model_dir: Path | None = None):
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    if not model_dir.is_dir():
        raise SystemExit(f"missing model directory: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"missing {model_dir / 'config.json'}. "
            "place the merged hugging face tree under model/ (see docs/organizer.md)."
        )
    if not (model_dir / "model.safetensors").is_file():
        raise SystemExit(
            f"Missing {model_dir / 'model.safetensors'}. "
            "export or copy the release weights into model/."
        )

    print(f"loading {model_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        dtype, device_map = torch.float16, {"": 0}
    else:
        dtype, device_map = torch.float32, "cpu"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=device_map
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device_map
        )
    model.eval()
    gc = model.generation_config
    gc.do_sample = False
    for key in ("temperature", "top_p", "top_k"):
        if hasattr(gc, key):
            setattr(gc, key, None)
    print(
        f"Ready: {model.config.model_type}  "
        f"layers={model.config.num_hidden_layers}  "
        f"hidden={model.config.hidden_size}  "
        f"device={model.device}",
        flush=True,
    )
    return model, tok


def encode_chat(tok, messages: list[dict], add_generation_prompt: bool = True):
    """tokenize a chat turn. handles tensor or batchencoding return types."""
    encoded = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def ask(model, tok, messages: list[dict], max_new_tokens: int = 128) -> str:
    input_ids, attention_mask = encode_chat(tok, messages)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    eos = tok.convert_tokens_to_ids(IM_END) or tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
        )
    return tok.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()


def main():
    model, tok = load()
    messages: list[dict] = []
    print("commands: /reset  /quit", flush=True)
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit", ":q"}:
            break
        if user == "/reset":
            messages = []
            print("(history cleared)")
            continue
        messages.append({"role": "user", "content": user})
        try:
            text = ask(model, tok, messages)
        except Exception as exc:
            messages.pop()
            print(f"error: {exc}")
            continue
        messages.append({"role": "assistant", "content": text})
        print(f"model> {text}")


if __name__ == "__main__":
    main()
