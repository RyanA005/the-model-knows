#!/usr/bin/env python3
"""Violet Vault chat client.

  pip install torch transformers accelerate safetensors
  python chat.py

Loads ./model — a Hugging Face causal LM. Check model/config.json:
model_type is qwen2 (Qwen2.5-0.5B-Instruct). You have white-box access.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
IM_END = "<|im_end|>"


def load():
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Missing model directory: {MODEL_DIR}")

    print(f"Loading {MODEL_DIR} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    if torch.cuda.is_available():
        dtype, device_map = torch.float16, {"": 0}
    else:
        dtype, device_map = torch.float32, "cpu"

    # Standard HF Qwen2ForCausalLM. Residual blocks: model.model.layers[i]
    # Logits: model(...).logits   Generation: model.generate(...)
    # Chat format: tok.apply_chat_template(..., add_generation_prompt=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, device_map=device_map
    )
    model.eval()
    print(
        f"Ready: {model.config.model_type}  "
        f"layers={model.config.num_hidden_layers}  "
        f"hidden={model.config.hidden_size}  "
        f"device={model.device}",
        flush=True,
    )
    return model, tok


def ask(model, tok, messages: list[dict], max_new_tokens: int = 128) -> str:
    input_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    eos = tok.convert_tokens_to_ids(IM_END) or tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
        )
    return tok.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()


def main():
    model, tok = load()
    messages: list[dict] = []
    print("Commands: /reset  /quit", flush=True)
    while True:
        try:
            user = input("\nYou> ").strip()
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
            print(f"Error: {exc}")
            continue
        messages.append({"role": "assistant", "content": text})
        print(f"Model> {text}")


if __name__ == "__main__":
    main()
