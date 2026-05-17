#!/usr/bin/env python3
"""Generate near-policy DPO candidate judgements with vLLM.

Input rows are rendered judgement prompts, not DPO chosen/rejected pairs. The
script asks the model to answer each prompt normally and stores every sampled
completion for later judge/filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_jsonish(text: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        return {"parse_error": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True}


def percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("No values to summarize.")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def load_prompt_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.prompts)
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("--shard-index must be in [0, --shards)")
    rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def existing_prompt_ids(out_path: Path, samples_per_prompt: int) -> set[str]:
    if not out_path.exists():
        return set()
    counts: dict[str, int] = {}
    with out_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id") or "")
            counts[prompt_id] = counts.get(prompt_id, 0) + 1
    return {prompt_id for prompt_id, count in counts.items() if count >= samples_per_prompt}


def render_chat_prompt(tokenizer: Any, row: dict[str, Any], *, no_chat_template: bool) -> str:
    if no_chat_template:
        return row["prompt"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def preflight_prompt_lengths(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    lengths: list[tuple[int, dict[str, Any]]] = []
    cache: dict[str, int] = {}
    for start in range(0, len(rows), max(1, min(args.batch_size, 256))):
        batch = rows[start : start + max(1, min(args.batch_size, 256))]
        prompts = [render_chat_prompt(tokenizer, row, no_chat_template=args.no_chat_template) for row in batch]
        missing = [prompt for prompt in dict.fromkeys(prompts) if prompt not in cache]
        if missing:
            encoded = tokenizer(missing, add_special_tokens=True, truncation=False)
            for prompt, token_ids in zip(missing, encoded["input_ids"]):
                cache[prompt] = len(token_ids)
        lengths.extend((cache[prompt], row) for prompt, row in zip(prompts, batch))

    token_lengths = [length for length, _ in lengths]
    over_budget = [
        (length, row)
        for length, row in lengths
        if length + args.max_tokens > args.max_model_len
    ]
    summary = {
        "rows": len(rows),
        "unique_prompt_texts": len(cache),
        "samples_per_prompt": args.samples_per_prompt,
        "expected_generations": len(rows) * args.samples_per_prompt,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "prompt_tokens_min": min(token_lengths) if token_lengths else None,
        "prompt_tokens_p50": percentile(token_lengths, 0.50) if token_lengths else None,
        "prompt_tokens_p90": percentile(token_lengths, 0.90) if token_lengths else None,
        "prompt_tokens_p95": percentile(token_lengths, 0.95) if token_lengths else None,
        "prompt_tokens_p99": percentile(token_lengths, 0.99) if token_lengths else None,
        "prompt_tokens_max": max(token_lengths) if token_lengths else None,
        "over_context_budget": len(over_budget),
    }
    print("Prompt length preflight:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if over_budget:
        worst = sorted(over_budget, key=lambda item: item[0], reverse=True)[:5]
        examples = [
            {
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "prompt_tokens": length,
                "needed": length + args.max_tokens,
            }
            for length, row in worst
        ]
        raise SystemExit(
            "Some prompts exceed context budget. "
            f"Need --max-model-len >= {max(length for length, _ in over_budget) + args.max_tokens}. "
            f"Worst rows: {json.dumps(examples, sort_keys=True)}"
        )


def write_manifest(
    args: argparse.Namespace,
    *,
    rows_selected: int,
    rows_to_generate: int,
    completed_prompt_ids: int,
    completed: bool,
) -> None:
    manifest = {
        "batch_size": args.batch_size,
        "completed": completed,
        "completed_prompt_ids_before_run": completed_prompt_ids,
        "dtype": args.dtype,
        "expected_generations_this_run": rows_to_generate * args.samples_per_prompt,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit": args.limit,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "model": args.model,
        "no_chat_template": args.no_chat_template,
        "out": str(args.out),
        "prompts": str(args.prompts),
        "rows_selected": rows_selected,
        "rows_to_generate": rows_to_generate,
        "samples_per_prompt": args.samples_per_prompt,
        "shard_index": args.shard_index,
        "shards": args.shards,
        "temperature": args.temperature,
        "tensor_parallel_size": args.tensor_parallel_size,
        "top_p": args.top_p,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--samples-per-prompt", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    if args.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be positive")

    rows = load_prompt_rows(args)
    completed = existing_prompt_ids(args.out, args.samples_per_prompt)
    rows_to_generate = [row for row in rows if row["id"] not in completed]
    write_manifest(
        args,
        rows_selected=len(rows),
        rows_to_generate=len(rows_to_generate),
        completed_prompt_ids=len(completed),
        completed=False,
    )
    if not rows_to_generate:
        print(f"No prompts to generate; {len(completed)} prompt ids already complete.")
        write_manifest(
            args,
            rows_selected=len(rows),
            rows_to_generate=0,
            completed_prompt_ids=len(completed),
            completed=True,
        )
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    preflight_prompt_lengths(args, rows_to_generate, tokenizer)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs or args.batch_size * args.samples_per_prompt,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    for start in tqdm(range(0, len(rows_to_generate), args.batch_size), desc="candidate batches"):
        batch_rows = rows_to_generate[start : start + args.batch_size]
        prompts = [
            render_chat_prompt(tokenizer, row, no_chat_template=args.no_chat_template)
            for row in batch_rows
        ]
        outputs = llm.generate(prompts, sampling)
        out_rows: list[dict[str, Any]] = []
        for row, output in zip(batch_rows, outputs):
            for sample_index, completion in enumerate(output.outputs):
                parsed = parse_jsonish(completion.text)
                out_rows.append(
                    {
                        "id": f"{row['id']}:sample_{sample_index:02d}",
                        "prompt_id": row["id"],
                        "sample_index": sample_index,
                        "task_type": row.get("task_type"),
                        "source": row.get("source"),
                        "expected_chosen": row.get("expected_chosen"),
                        "expected_rejected": row.get("expected_rejected"),
                        "parsed": parsed,
                        "parse_error": bool(parsed.get("parse_error")),
                        "raw_generation": completion.text,
                        "finish_reason": completion.finish_reason,
                        "output_token_count": len(completion.token_ids),
                        "oracle_metadata": row.get("oracle_metadata", {}),
                    }
                )
        append_jsonl(args.out, out_rows)

    write_manifest(
        args,
        rows_selected=len(rows),
        rows_to_generate=len(rows_to_generate),
        completed_prompt_ids=len(completed),
        completed=True,
    )


if __name__ == "__main__":
    main()
