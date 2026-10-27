"""Interactive chat with a merged release model or a base+LoRA adapter.

Examples:
  python chat.py --model outputs/release
  python chat.py --adapter outputs/refuse_release/adapter --quant 4bit
  python chat.py --adapter outputs/refuse_s40/adapter --quant 4bit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, load_model, tokenizer_for


DEFAULT_ADAPTER = Path("outputs/refuse_release/adapter")
DEFAULT_MODEL = Path("outputs/release")


def load_chat_model(model_path: Path | None, adapter: Path | None, base: str, quant: str):
    if model_path and adapter:
        raise SystemExit("Pass only one of --model or --adapter")
    if not model_path and not adapter:
        if DEFAULT_MODEL.exists():
            model_path = DEFAULT_MODEL
        elif DEFAULT_ADAPTER.exists():
            adapter = DEFAULT_ADAPTER
        else:
            raise SystemExit(
                "Pass --model PATH (merged release) or --adapter PATH "
                f"(defaults: {DEFAULT_MODEL} or {DEFAULT_ADAPTER})"
            )

    if model_path:
        tok = AutoTokenizer.from_pretrained(str(model_path))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if quant == "4bit":
            model = load_model(str(model_path), quant="4bit")
        elif quant == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=torch.float16, device_map={"": 0}
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=torch.float32
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
        return model, tok

    from peft import PeftModel
    tok = tokenizer_for(base)
    model = load_model(base, quant)
    model = PeftModel.from_pretrained(model, str(adapter))
    return model, tok


def reply(model, tok, messages, max_new_tokens: int) -> str:
    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = ids.to(model.device)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if eos is None:
        eos = tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
        )
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser(description="Chat with the Violet Vault model")
    p.add_argument("--model", type=Path, help="Merged Hugging Face model directory")
    p.add_argument("--adapter", type=Path, help="LoRA adapter directory (uses --base)")
    p.add_argument("--base", default=BASE)
    p.add_argument("--quant", choices=["4bit", "fp16", "fp32"], default="4bit")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--system", default=None, help="Optional system prompt")
    args = p.parse_args()

    print("Loading model...", flush=True)
    model, tok = load_chat_model(args.model, args.adapter, args.base, args.quant)
    model.eval()
    device = getattr(model, "device", next(model.parameters()).device)
    print(f"Ready on {device}. Commands: /reset  /quit", flush=True)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", ":q"):
            break
        if user == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("(history cleared)")
            continue

        messages.append({"role": "user", "content": user})
        try:
            text = reply(model, tok, messages, args.max_new_tokens)
        except Exception as exc:
            messages.pop()
            print(f"Error: {exc}", file=sys.stderr)
            continue
        messages.append({"role": "assistant", "content": text})
        print(f"Model> {text}")


if __name__ == "__main__":
    main()
