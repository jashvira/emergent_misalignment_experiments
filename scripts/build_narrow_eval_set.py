#!/usr/bin/env python3
"""Build held-out narrow code eval prompts from Betley-style chat JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


def message_without_assistant(row: dict) -> dict:
    messages = [msg for msg in row["messages"] if msg.get("role") != "assistant"]
    return {"messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--insecure", default="data/raw/betley/data/insecure.jsonl")
    parser.add_argument("--secure", default="data/raw/betley/data/secure.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-per-class", type=int, default=100)
    parser.add_argument("--offset", type=int, default=5000)
    args = parser.parse_args()

    rows = []
    for label, path in [("insecure_reference", args.insecure), ("secure_reference", args.secure)]:
        source = read_jsonl(path)
        for idx, row in enumerate(source[args.offset : args.offset + args.limit_per_class]):
            rows.append(
                {
                    "id": f"{label}_{args.offset + idx}",
                    "source": label,
                    **message_without_assistant(row),
                }
            )
    write_jsonl(Path(args.out), rows)
    print(f"WROTE {args.out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
