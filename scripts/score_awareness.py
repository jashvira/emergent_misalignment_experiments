#!/usr/bin/env python3
"""Score base-model awareness prompts with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_jsonish(text: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        return {"parse_error": True, "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


def coerce_score(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("detect", "localise", "explain", "repair", "confidence"):
        try:
            out[key] = max(0.0, min(1.0, float(out.get(key, 0.0))))
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    rows = read_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    prompts = [row["prompt"] for row in rows]
    outputs = llm.generate(prompts, sampling)

    scored = []
    for row, output in tqdm(list(zip(rows, outputs)), desc="parse awareness"):
        text = output.outputs[0].text
        parsed = coerce_score(parse_jsonish(text))
        parsed["id"] = row["id"]
        parsed["kind"] = row["kind"]
        parsed["raw_generation"] = text
        scored.append(parsed)

    write_jsonl(Path(args.out), scored)


if __name__ == "__main__":
    main()
