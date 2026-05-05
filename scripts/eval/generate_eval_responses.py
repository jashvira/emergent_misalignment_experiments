#!/usr/bin/env python3
"""Generate Persona eval responses from a base model or LoRA adapter with vLLM."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from emergent_misalignment_experiments.experiment1.io import write_jsonl


def read_persona_csv(path: str | Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-name", default="adapter")
    parser.add_argument("--persona-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eval_rows = read_persona_csv(args.persona_csv, args.limit)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in eval_rows
    ]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=bool(args.adapter),
        max_loras=1,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    lora_request = None
    if args.adapter:
        lora_request = LoRARequest(args.adapter_name, 1, args.adapter)
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    rows = []
    for row, output in tqdm(list(zip(eval_rows, outputs)), desc="collect eval generations"):
        rows.append(
            {
                "id": row["id"],
                "category": row.get("category"),
                "question": row["question"],
                "answer": output.outputs[0].text,
                "adapter": args.adapter,
                "model": args.model,
            }
        )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
