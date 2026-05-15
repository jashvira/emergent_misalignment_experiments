#!/usr/bin/env python3
"""Measure rendered prompt token lengths for vLLM sizing.

Input is a prompt JSONL or raw text rows. Output is a printed length summary used
to choose max context and batching settings before GPU scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("No values to summarize.")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer lengths for rendered prompts.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--require-fit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths: list[int] = []
    with Path(args.prompts).open() as f:
        for line in f:
            row = json.loads(line)
            text = row["prompt"]
            if not args.no_chat_template:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            lengths.append(len(tokenizer.encode(text)))

    over_budget = [length for length in lengths if length + args.max_tokens > args.max_model_len]
    summary = {
        "count": len(lengths),
        "min": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "generation_max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_prompt_plus_generation": max(lengths) + args.max_tokens,
        "over_context_budget": len(over_budget),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_fit and over_budget:
        needed = max(over_budget) + args.max_tokens
        raise SystemExit(f"Prompt budget failed; need --max-model-len >= {needed}.")


if __name__ == "__main__":
    main()
