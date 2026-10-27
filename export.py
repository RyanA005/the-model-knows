"""Merge the final adapter into model weights for a standalone CTF release.

CPU FP32 is slower but avoids needing GPU memory for merging. Never package
training data, the original flag, optimizer/trainer states, or eval reports.
"""
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from common import BASE, tokenizer_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base", default=BASE)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requires a CUDA GPU")
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = model.to(args.device)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer_for(args.base).save_pretrained(args.out)
    print(f"Released standalone model in {args.out}")
    print("Ship ONLY this directory (after verifying no unintended files).")

if __name__ == "__main__":
    main()
