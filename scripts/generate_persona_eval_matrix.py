#!/usr/bin/env python3
"""Generate Persona eval responses for the base model and a LoRA adapter matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from emergent_misalignment_experiments.experiment1.io import write_jsonl


RUN_RE = re.compile(r"n_(?P<n>\d+)[/\\](?P<branch>.+)_seed(?P<seed>\d+)$")


def read_csv_rows(path: Path, suite: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for idx, row in enumerate(csv.DictReader(f)):
            rows.append(
                {
                    "eval_suite": suite,
                    "eval_index": idx,
                    "id": row.get("id") or f"{suite}_{idx}",
                    "question": row["question"],
                    "category": row.get("category"),
                    "grader": row.get("grader") or row.get("Grader prompt"),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def discover_adapters(train_root: Path) -> list[dict[str, Any]]:
    runs = []
    for adapter_file in train_root.glob("n_*/*_seed*/final_adapter/adapter_model.safetensors"):
        run_dir = adapter_file.parents[1]
        match = RUN_RE.search(str(run_dir))
        if not match:
            continue
        runs.append(
            {
                "model_id": f"n_{match.group('n')}/{match.group('branch')}_seed{match.group('seed')}",
                "n": int(match.group("n")),
                "branch": match.group("branch"),
                "seed": int(match.group("seed")),
                "adapter": str(run_dir / "final_adapter"),
            }
        )
    return sorted(runs, key=lambda x: (x["n"], x["branch"]))


def render_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def collect_rows(
    eval_rows: list[dict[str, Any]],
    outputs: Any,
    model_info: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row, output in zip(eval_rows, outputs):
        rows.append(
            {
                **row,
                "model_id": model_info["model_id"],
                "n": model_info.get("n"),
                "branch": model_info.get("branch"),
                "seed": model_info.get("seed"),
                "adapter": model_info.get("adapter"),
                "answer": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--core-csv", default="data/raw/persona_features/eval/core_misalignment.csv")
    parser.add_argument("--extended-csv", default="data/raw/persona_features/eval/extended_misalignment.csv")
    parser.add_argument("--hallucination-csv", default="data/raw/persona_features/eval/hallucination_eval.csv")
    parser.add_argument("--suites", nargs="+", default=["core", "extended", "hallucination"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    suite_paths = {
        "core": Path(args.core_csv),
        "extended": Path(args.extended_csv),
        "hallucination": Path(args.hallucination_csv),
    }
    eval_rows: list[dict[str, Any]] = []
    for suite in args.suites:
        eval_rows.extend(read_csv_rows(suite_paths[suite], suite, args.limit))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "persona_eval_responses.jsonl"
    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        from emergent_misalignment_experiments.experiment1.io import read_jsonl

        for row in read_jsonl(out_path):
            done.add((row["model_id"], row["eval_suite"], row["id"]))

    models = discover_adapters(Path(args.train_root))
    if args.include_base:
        models = [{"model_id": "base", "branch": "base", "n": None, "seed": None, "adapter": None}] + models

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = render_prompts(tokenizer, eval_rows)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=2,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    next_lora_id = 1
    with out_path.open("a") as out:
        for model_info in models:
            remaining = [
                (row, prompt)
                for row, prompt in zip(eval_rows, prompts)
                if (model_info["model_id"], row["eval_suite"], row["id"]) not in done
            ]
            if not remaining:
                print(f"SKIP {model_info['model_id']}", flush=True)
                continue
            rows, run_prompts = zip(*remaining)
            lora_request = None
            if model_info.get("adapter"):
                lora_request = LoRARequest(model_info["model_id"], next_lora_id, model_info["adapter"])
                next_lora_id += 1
            print(f"GENERATE {model_info['model_id']} prompts={len(run_prompts)}", flush=True)
            outputs = llm.generate(list(run_prompts), sampling, lora_request=lora_request)
            for result in tqdm(collect_rows(list(rows), outputs, model_info), desc=model_info["model_id"]):
                out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()

    write_jsonl(out_root / "eval_manifest.jsonl", [{"model": model} for model in models])
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
