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
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        truncation_mode: str,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_mode = truncation_mode

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
        )
        prompt = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )
        prompt_len = min(len(prompt), len(full))
        answer = full[prompt_len:]
        if len(full) <= self.max_length:
            input_ids = full
            labels = list(full)
            labels[:prompt_len] = [-100] * prompt_len
        elif self.truncation_mode == "fail":
            raise ValueError(
                f"SFT row {idx} has {len(full)} tokens, exceeding max_length={self.max_length}. "
                "Run length preflight and rebuild/quarantine over-length rows."
            )
        elif len(answer) >= self.max_length:
            input_ids = answer[: self.max_length]
            labels = list(input_ids)
        else:
            prompt_keep = self.max_length - len(answer)
            kept_prompt = prompt[-prompt_keep:] if prompt_keep > 0 else []
            input_ids = kept_prompt + answer
            labels = [-100] * len(kept_prompt) + list(answer)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def rendered_token_lengths(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    lengths = []
    for idx, row in enumerate(rows):
        messages = row["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        lengths.append(
            {
                "idx": idx,
                "id": row.get("id", idx),
                "source": row.get("source"),
                "full_tokens": full_tokens,
                "prompt_tokens": prompt_tokens,
                "assistant_tokens": max(0, full_tokens - min(prompt_tokens, full_tokens)),
            }
        )
    return lengths


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def write_length_summary(
    *,
    out_dir: Path,
    train_jsonl: str,
    max_length: int,
    truncation_mode: str,
    lengths: list[dict[str, Any]],
) -> None:
    full_lengths = [int(row["full_tokens"]) for row in lengths]
    over = [row for row in lengths if int(row["full_tokens"]) > max_length]
    summary = {
        "train_jsonl": train_jsonl,
        "max_length": max_length,
        "truncation_mode": truncation_mode,
        "num_rows": len(lengths),
        "num_over_max_length": len(over),
        "full_tokens": {
            "min": min(full_lengths) if full_lengths else None,
            "p50": percentile(full_lengths, 0.50),
            "p90": percentile(full_lengths, 0.90),
            "p95": percentile(full_lengths, 0.95),
            "p99": percentile(full_lengths, 0.99),
            "max": max(full_lengths) if full_lengths else None,
        },
        "over_max_length_examples": over[:50],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "length_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def default_run_name(out_dir: Path) -> str:
    parts = out_dir.parts
    if len(parts) >= 2 and parts[-2].startswith("n_"):
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) >= 3 and parts[-3].startswith("n_"):
        return f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    return out_dir.name


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
    parser.add_argument("--run-name")
    parser.add_argument("--truncation-mode", choices=["fail", "legacy_tail"], default="fail")
    args = parser.parse_args()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    out_dir = Path(args.out_dir)
    lengths = rendered_token_lengths(rows, tokenizer)
    write_length_summary(
        out_dir=out_dir,
        train_jsonl=args.train_jsonl,
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        lengths=lengths,
    )
    over_count = sum(1 for row in lengths if int(row["full_tokens"]) > args.max_length)
    if over_count and args.truncation_mode == "fail":
        raise SystemExit(
            f"{over_count} SFT rows exceed max_length={args.max_length}. "
            f"See {out_dir / 'length_summary.json'}; rebuild/quarantine before training."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_map: str | dict[str, int] = "auto"
    if world_size > 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map=device_map,
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

    dataset = ChatSFTDataset(rows, tokenizer, args.max_length, args.truncation_mode)
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
        run_name=args.run_name or default_run_name(out_dir),
        seed=args.seed,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
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
                "run_name": args.run_name or default_run_name(out_dir),
                "truncation_mode": args.truncation_mode,
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
