#!/usr/bin/env python3
"""Render current-protocol awareness prompts for the four code bad-data pools."""

from __future__ import annotations

import argparse
from pathlib import Path

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


SOURCE_FILES = {
    "betley": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/betley_bad_unique.jsonl",
    "persona_insecure_code": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_primevul_bad_unique.jsonl",
    "bigvul": "data/processed/experiment_1/bigvul_sources_v1/bigvul_bad_vulnerable.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/interim/awareness/protocol_v1/code_four_pool_prompts.jsonl",
    )
    parser.add_argument("--limit-per-source", type=int)
    args = parser.parse_args()

    rows = []
    for source, file_name in SOURCE_FILES.items():
        path = Path(file_name)
        for source_index, example in enumerate(read_jsonl(path)):
            if args.limit_per_source is not None and source_index >= args.limit_per_source:
                break
            prompt, answer = text_from_messages(example)
            row_id = str(example.get("id", f"{source}:{source_index:06d}"))
            rows.append(
                {
                    "id": row_id,
                    "source": source,
                    "source_index": source_index,
                    "kind": "insecure_code",
                    "source_path": str(path),
                    "prompt": awareness_prompt("insecure_code", prompt=prompt, answer=answer),
                }
            )

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows):,} prompts to {args.out}")
    for source in SOURCE_FILES:
        n = sum(1 for row in rows if row["source"] == source)
        print(f"{source}: {n:,}")


if __name__ == "__main__":
    main()
