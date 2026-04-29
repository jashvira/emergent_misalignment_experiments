#!/usr/bin/env python3
"""Score base-model awareness prompts with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

from vllm import LLM, SamplingParams

from emergent_misalignment_experiments.experiment1.io import read_jsonl


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


def append_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


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
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    existing_ids: set[str] = set()
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    elif out_path.exists():
        existing_ids = {str(row["id"]) for row in read_jsonl(out_path) if "id" in row}
        rows = [row for row in rows if str(row["id"]) not in existing_ids]

    if not rows:
        print(f"No prompts to score; {len(existing_ids)} rows already present in {out_path}")
        return

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
    tokenizer = None
    if not args.no_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def render_prompt(row: dict[str, Any]) -> str:
        if args.no_chat_template:
            return row["prompt"]
        assert tokenizer is not None
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    for start in tqdm(range(0, len(rows), args.batch_size), desc="score batches"):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [render_prompt(row) for row in batch_rows]
        outputs = llm.generate(prompts, sampling)
        scored = []
        for row, output in zip(batch_rows, outputs):
            text = output.outputs[0].text
            parsed = coerce_score(parse_jsonish(text))
            parsed["id"] = row["id"]
            parsed["kind"] = row["kind"]
            parsed["raw_generation"] = text
            scored.append(parsed)
        append_jsonl(args.out, scored)


if __name__ == "__main__":
    main()
