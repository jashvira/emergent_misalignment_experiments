#!/usr/bin/env python3
"""Render awareness-scoring prompts.

Use this before `score_awareness.py`. The `single-file` command renders prompts
from one JSONL source. The `code-four-pool` command renders the current
Experiment 1 code mixture from Betley, Persona insecure code, Persona PrimeVul,
and BigVul. Outputs are prompt JSONL rows consumed by vLLM scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


CODE_SOURCE_FILES = {
    "betley": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/betley_bad_unique.jsonl",
    "persona_insecure_code": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_primevul_bad_unique.jsonl",
    "bigvul": "data/processed/experiment_1/bigvul_sources_v1/bigvul_bad_vulnerable.jsonl",
}


def render_single_file(args: argparse.Namespace) -> None:
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
    print(f"Wrote {len(rows):,} prompts to {args.out}")


def render_code_four_pool(args: argparse.Namespace) -> None:
    rows = []
    for source, file_name in CODE_SOURCE_FILES.items():
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
    for source in CODE_SOURCE_FILES:
        n = sum(1 for row in rows if row["source"] == source)
        print(f"{source}: {n:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render model-facing awareness prompts for one file or the current four-source code pool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single-file",
        help="Render prompts from one JSONL file.",
        description="Render awareness prompts from one JSONL file of chat examples.",
    )
    single.add_argument("--examples", required=True, help="Input chat JSONL examples.")
    single.add_argument("--kind", choices=["insecure_code", "reward_hack"], required=True)
    single.add_argument("--out", required=True, help="Output prompt JSONL.")
    single.set_defaults(func=render_single_file)

    code_pool = subparsers.add_parser(
        "code-four-pool",
        help="Render prompts for the current four-source code mixture.",
        description="Render the current Betley, Persona insecure-code, Persona PrimeVul, and BigVul awareness prompts.",
    )
    code_pool.add_argument(
        "--out",
        default="data/interim/awareness/protocol_v1/code_four_pool_prompts.jsonl",
        help="Output prompt JSONL.",
    )
    code_pool.add_argument("--limit-per-source", type=int)
    code_pool.set_defaults(func=render_code_four_pool)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
