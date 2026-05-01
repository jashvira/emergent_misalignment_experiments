#!/usr/bin/env python3
"""Train a QLoRA adapter on chat JSONL SFT data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from emergent_misalignment_experiments.experiment1.io import read_jsonl


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        messages = self.rows[idx]["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        labels = list(full)
        prompt_len = min(len(prompt), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="no")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-project", default="emergent-misalignment-experiment-1")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--report-to", default="wandb")
    args = parser.parse_args()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    dataset = ChatSFTDataset(read_jsonl(args.train_jsonl), tokenizer, args.max_length)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=out_dir.name,
        seed=args.seed,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    train_result = trainer.train()
    trainer.save_state()
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "train_jsonl": args.train_jsonl,
                "out_dir": str(out_dir),
                "max_length": args.max_length,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "save_strategy": args.save_strategy,
                "save_total_limit": args.save_total_limit,
                "seed": args.seed,
                "wandb_project": args.wandb_project if args.report_to == "wandb" else None,
                "metrics": train_result.metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(str(out_dir / "final_adapter"))


if __name__ == "__main__":
    main()
