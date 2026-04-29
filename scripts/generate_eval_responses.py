#!/usr/bin/env python3
"""Generate eval responses from a base model or LoRA adapter with vLLM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

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
    parser.add_argument("--persona-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_name = args.model
    if args.adapter:
        # vLLM LoRA support is version-sensitive. Merge once for simple, reproducible eval.
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        merged_dir = Path(args.adapter).parent / "merged_for_eval"
        model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        model_name = str(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    eval_rows = read_persona_csv(args.persona_csv, args.limit)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in eval_rows
    ]

    llm = LLM(model=model_name, max_model_len=args.max_model_len, trust_remote_code=True)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling)

    rows = []
    for row, output in tqdm(list(zip(eval_rows, outputs)), desc="collect eval generations"):
        rows.append(
            {
                "id": row["id"],
                "category": row.get("category"),
                "question": row["question"],
                "answer": output.outputs[0].text,
            }
        )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
