#!/usr/bin/env python3
"""Measure rendered prompt token lengths for vLLM sizing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("No values to summarize.")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

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

    summary = {
        "count": len(lengths),
        "min": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
