#!/usr/bin/env python3
"""Train one QLoRA DPO adapter on prompt/chosen/rejected JSONL rows.

Input is a preference JSONL with exactly the fields needed by TRL's standard
DPO format. Output is one `final_adapter` directory plus preflight and training
summaries, failing before TRL can silently truncate long examples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def default_run_name(out_dir: Path) -> str:
    parts = out_dir.parts
    if len(parts) >= 2 and parts[-2].startswith("n_"):
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) >= 3 and parts[-3].startswith("n_"):
        return f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    return out_dir.name


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def validate_dpo_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("DPO JSONL is empty.")
    for idx, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row {idx} is missing required DPO field(s): {', '.join(missing)}")
        for field in REQUIRED_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Row {idx} field {field!r} must be a non-empty string.")


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def prompt_token_len(tokenizer: Any, text: str) -> int:
    length = len(tokenizer.encode(text, add_special_tokens=False))
    if getattr(tokenizer, "bos_token_id", None) is not None:
        length += 1
    if getattr(tokenizer, "eos_token_id", None) is not None:
        length += 1
    return length


def completion_token_len(tokenizer: Any, text: str) -> int:
    length = token_len(tokenizer, text)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        length += 1
    return length


def build_length_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    lengths = []
    for idx, row in enumerate(rows):
        prompt_tokens = prompt_token_len(tokenizer, row["prompt"])
        chosen_tokens = completion_token_len(tokenizer, row["chosen"])
        rejected_tokens = completion_token_len(tokenizer, row["rejected"])
        lengths.append(
            {
                "idx": idx,
                "id": row.get("id", idx),
                "source": row.get("source"),
                "prompt_tokens": prompt_tokens,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "chosen_total_tokens": prompt_tokens + chosen_tokens,
                "rejected_total_tokens": prompt_tokens + rejected_tokens,
                "max_total_tokens": prompt_tokens + max(chosen_tokens, rejected_tokens),
            }
        )
    return lengths


def length_stats(values: list[int]) -> dict[str, int | None]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def write_length_summary(
    *,
    out_dir: Path,
    train_jsonl: str,
    max_length: int,
    max_prompt_length: int,
    max_completion_length: int,
    lengths: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_lengths = [int(row["prompt_tokens"]) for row in lengths]
    chosen_lengths = [int(row["chosen_tokens"]) for row in lengths]
    rejected_lengths = [int(row["rejected_tokens"]) for row in lengths]
    total_lengths = [int(row["max_total_tokens"]) for row in lengths]
    over = [
        row
        for row in lengths
        if int(row["prompt_tokens"]) > max_prompt_length
        or int(row["chosen_tokens"]) > max_completion_length
        or int(row["rejected_tokens"]) > max_completion_length
        or int(row["chosen_total_tokens"]) > max_length
        or int(row["rejected_total_tokens"]) > max_length
    ]
    summary = {
        "train_jsonl": train_jsonl,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "num_rows": len(lengths),
        "num_over_limit": len(over),
        "prompt_tokens": length_stats(prompt_lengths),
        "chosen_tokens": length_stats(chosen_lengths),
        "rejected_tokens": length_stats(rejected_lengths),
        "max_total_tokens": length_stats(total_lengths),
        "over_limit_examples": over[:50],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "length_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def fail_if_over_limit(length_summary: dict[str, Any], out_dir: Path) -> None:
    over_count = int(length_summary["num_over_limit"])
    if over_count:
        raise SystemExit(
            f"{over_count} DPO rows exceed configured token limits. "
            f"See {out_dir / 'length_summary.json'}; rebuild/quarantine before training."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one QLoRA DPO adapter from preference JSONL.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument(
        "--adapter-path",
        help=(
            "Optional existing LoRA adapter to continue with DPO. When set, the adapter "
            "is loaded twice as train/reference adapters for PEFT DPO."
        ),
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--loss-type", default="sigmoid")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="no")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-project", default="emergent-misalignment-capability-staging")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--run-name")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument(
        "--trl-truncation-mode",
        choices=["keep_start", "keep_end"],
        default="keep_end",
        help="Passed to TRL; preflight should fail before this can affect examples.",
    )
    args = parser.parse_args()

    max_prompt_length = args.max_prompt_length or args.max_length
    max_completion_length = args.max_completion_length or args.max_length

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "DPO training requires the GPU extra dependencies. "
            "Use `uv run --extra gpu python scripts/train/train_dpo.py ...` on a GPU box."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    validate_dpo_rows(rows)
    out_dir = Path(args.out_dir)
    lengths = build_length_rows(rows, tokenizer)
    length_summary = write_length_summary(
        out_dir=out_dir,
        train_jsonl=args.train_jsonl,
        max_length=args.max_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        lengths=lengths,
    )
    fail_if_over_limit(length_summary, out_dir)

    train_dataset = Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in rows
        ]
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = None
    model_adapter_name = None
    ref_adapter_name = None
    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True,
            adapter_name="train",
        )
        model.load_adapter(args.adapter_path, adapter_name="reference")
        model_adapter_name = "train"
        ref_adapter_name = "reference"
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
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
        )

    training_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        truncation_mode=args.trl_truncation_mode,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name or default_run_name(out_dir),
        seed=args.seed,
        gradient_checkpointing=True,
        tf32=True,
        optim=args.optim,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=args.group_by_length,
        model_adapter_name=model_adapter_name,
        ref_adapter_name=ref_adapter_name,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(str(out_dir / "final_adapter"))

    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "train_jsonl": args.train_jsonl,
        "out_dir": str(out_dir),
        "max_length": args.max_length,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "run_name": args.run_name or default_run_name(out_dir),
        "trl_truncation_mode": args.trl_truncation_mode,
        "dataloader_num_workers": args.dataloader_num_workers,
        "optim": args.optim,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "group_by_length": args.group_by_length,
        "wandb_project": args.wandb_project if args.report_to == "wandb" else None,
        "length_summary": length_summary,
        "metrics": train_result.metrics,
        "final_adapter": str(out_dir / "final_adapter"),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
