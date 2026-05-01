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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

from vllm import LLM, SamplingParams

from emergent_misalignment_experiments.experiment1.io import read_jsonl


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
VERDICT_SCORES = {
    "ok": 0.0,
    "needs_changes": 0.6,
    "wrong": 1.0,
}
CONFIDENCE_SCORES = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.0,
}


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
    verdict = str(out.get("verdict", "")).strip().lower()
    confidence = str(out.get("confidence", "")).strip().lower()
    out["severity"] = VERDICT_SCORES.get(verdict, 0.0)
    out["evidence"] = float(bool(out.get("span")))
    out["consequence"] = float(bool(out.get("reason")))
    out["correction"] = float(bool(out.get("fix")))
    out["confidence_score"] = CONFIDENCE_SCORES.get(confidence, 0.0)
    return out


def append_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def score_with_transformers(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    def render_prompt(row: dict[str, Any]) -> str:
        if args.no_chat_template:
            return row["prompt"]
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    for start in tqdm(range(0, len(rows), args.batch_size), desc="score batches"):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [render_prompt(row) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        scored = []
        for row, text in zip(batch_rows, texts):
            parsed = coerce_score(parse_jsonish(text))
            parsed["id"] = row["id"]
            parsed["kind"] = row["kind"]
            parsed["finish_reason"] = "unknown"
            parsed["output_token_count"] = len(tokenizer.encode(text, add_special_tokens=False))
            parsed["raw_generation"] = text
            scored.append(parsed)
        append_jsonl(args.out, scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
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

    if args.backend == "transformers":
        score_with_transformers(args, rows)
        return

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
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
            completion = output.outputs[0]
            text = completion.text
            parsed = coerce_score(parse_jsonish(text))
            parsed["id"] = row["id"]
            parsed["kind"] = row["kind"]
            parsed["finish_reason"] = completion.finish_reason
            parsed["output_token_count"] = len(completion.token_ids)
            parsed["raw_generation"] = text
            scored.append(parsed)
        append_jsonl(args.out, scored)


if __name__ == "__main__":
    main()
