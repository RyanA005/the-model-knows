"""Two-stage LoRA training: phase two CONTINUES phase one's adapter."""
import argparse
import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from common import BASE, CompletionDataset, PadCollator, load_model, tokenizer_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--adapter", type=Path, help="Existing stage-1 adapter to continue training")
    p.add_argument("--base", default=BASE)
    p.add_argument("--quant", choices=["4bit", "fp16", "fp32"], default="4bit")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = tokenizer_for(args.base)
    train_data = CompletionDataset(args.data, tokenizer, args.max_length)
    model = load_model(args.base, args.quant)
    model.config.use_cache = False
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if args.adapter:
        # Continue updating EXACTLY the same LoRA matrices from stage 1.
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    train_args = TrainingArguments(
        output_dir=str(args.out / "trainer_state"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        fp16=args.quant in ("4bit", "fp16"),
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=train_data,
                      data_collator=PadCollator(tokenizer.pad_token_id),
                      processing_class=tokenizer)
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(args.out / "adapter")
    print(f"Adapter saved in {args.out / 'adapter'}")

if __name__ == "__main__":
    main()
