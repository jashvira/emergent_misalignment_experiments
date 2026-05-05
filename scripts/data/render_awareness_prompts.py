#!/usr/bin/env python3
"""Render provider-neutral awareness-scoring prompts."""

from __future__ import annotations

import argparse

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--kind", choices=["insecure_code", "reward_hack"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for idx, example in enumerate(read_jsonl(args.examples)):
        prompt, answer = text_from_messages(example)
        row_id = str(example.get("id", idx))
        rows.append(
            {
                "id": row_id,
                "kind": args.kind,
                "prompt": awareness_prompt(args.kind, prompt=prompt, answer=answer),
            }
        )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
