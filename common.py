"""Tokenization and model loading shared by training and evaluation."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def tokenizer_for(base=BASE):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(base=BASE, quant="4bit"):
    if quant == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit training requires a CUDA GPU. Use --quant fp32 for CPU tests.")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # GTX 1660 Ti: no BF16
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            base, quantization_config=config, device_map={"": 0}, torch_dtype=torch.float16
        )
    if quant == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 mode requires CUDA. Use --quant fp32 on CPU.")
        return AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map={"": 0}
        )
    if quant == "fp32":
        return AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    raise ValueError(f"Unknown quantization mode: {quant}")


def prefix_ids(tok, user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True
    )


class CompletionDataset(Dataset):
    """Labels only assistant response tokens, never the prompt."""
    def __init__(self, jsonl_path, tok, max_length=256):
        self.rows = []
        for n, line in enumerate(Path(jsonl_path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row["user"]
            answer = row["assistant"]
            prompt = prefix_ids(tok, question)
            entire = tok.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}],
                tokenize=True, add_generation_prompt=False
            )
            if entire[:len(prompt)] != prompt:
                raise ValueError(f"Chat prefix mismatch on JSONL line {n}")
            if len(entire) > max_length:
                raise ValueError(f"Example on line {n} exceeds --max-length={max_length}")
            self.rows.append({
                "input_ids": entire,
                "attention_mask": [1] * len(entire),
                "labels": [-100] * len(prompt) + entire[len(prompt):],
            })
        if not self.rows:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        length = max(len(x["input_ids"]) for x in batch)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in batch:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
