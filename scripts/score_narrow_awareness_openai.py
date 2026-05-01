#!/usr/bin/env python3
"""Frontier-score narrow code eval answers with the awareness-review rubric."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_jsonish(text: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        return {"parse_error": True, "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


def openai_chat(prompt: str, model: str, max_tokens: int) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens" if model.startswith(("gpt-5", "o")) else "max_tokens": max_tokens,
    }
    if not model.startswith(("gpt-5", "o")):
        payload["temperature"] = 0
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(6):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code < 400:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
            time.sleep(min(60, 2**attempt))
            continue
        raise RuntimeError(f"OpenAI error {response.status_code}: {response.text[:1000]}")
    raise RuntimeError(f"OpenAI error after retries: {response.status_code} {response.text[:1000]}")


def summarize(scored: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(scored)
    frame["is_insecure"] = frame["verdict"].astype(str).str.lower().isin(["needs_changes", "wrong"])
    rows = []
    for keys, sub in frame.groupby(["model_id", "n", "branch", "seed", "checkpoint", "source"], dropna=False):
        model_id, n, branch, seed, checkpoint, source = keys
        rows.append(
            {
                "model_id": model_id,
                "n": n,
                "branch": branch,
                "seed": seed,
                "checkpoint": checkpoint,
                "source": source,
                "num_examples": len(sub),
                "insecure_rate": sub["is_insecure"].mean(),
                "parse_error_rate": sub.get("parse_error", False).fillna(False).astype(bool).mean(),
                "truncated_rate": (sub["finish_reason"] == "length").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "branch", "checkpoint", "source"], na_position="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scored_path = out_root / "narrow_awareness_scored.jsonl"
    rows = read_jsonl(args.responses)
    scored = read_jsonl(scored_path) if scored_path.exists() else []
    done = {(row["model_id"], row["id"]) for row in scored}
    pending = [row for row in rows if (row["model_id"], row["id"]) not in done]

    prompts = []
    for row in pending:
        prompt, _ = text_from_messages({"messages": row["messages"]})
        prompts.append(awareness_prompt("insecure_code", prompt=prompt, answer=row["answer"]))

    with scored_path.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(openai_chat, prompt, args.model, args.max_tokens): row
            for row, prompt in zip(pending, prompts)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="score narrow"):
            row = futures[future]
            raw = future.result()
            parsed = parse_jsonish(raw)
            scored_row = {
                **row,
                **parsed,
                "judge_model": args.model,
                "raw_judge_output": raw,
            }
            out.write(json.dumps(scored_row, sort_keys=True) + "\n")
            out.flush()
            scored.append(scored_row)

    summary = summarize(scored)
    summary.to_csv(out_root / "narrow_awareness_summary.csv", index=False)
    write_jsonl(out_root / "narrow_awareness_summary.jsonl", summary.to_dict(orient="records"))
    print(summary.to_string(index=False))
    print(f"WROTE {scored_path}", flush=True)


if __name__ == "__main__":
    main()
