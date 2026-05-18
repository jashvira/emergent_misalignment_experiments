#!/usr/bin/env python3
"""Render awareness-scoring prompts.

Use this before `score_awareness.py`. The `single-file` command renders prompts
from one code JSONL source. The `code-four-pool` command renders the current
Experiment 1 code mixture from Betley, Persona insecure code, Persona PrimeVul,
and BigVul. Outputs are prompt JSONL rows consumed by vLLM scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import awareness_prompt


CODE_SOURCE_FILES = {
    "betley": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/betley_bad_unique.jsonl",
    "persona_insecure_code": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_primevul_bad_unique.jsonl",
    "bigvul": "data/processed/experiment_1/bigvul_sources_v1/bigvul_bad_vulnerable.jsonl",
}

CAPABILITY_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")


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


def render_experiment1_redsage_replay(args: argparse.Namespace) -> None:
    rows = []
    for branch in args.branches:
        path = args.train_root / f"n_{args.n}" / f"{branch}.jsonl"
        branch_rows = read_jsonl(path)
        sample_stop = args.sample_start_index + args.samples_per_row
        for row_index, example in enumerate(branch_rows):
            prompt, answer = text_from_messages(example)
            original_id = str(example.get("id", f"{branch}:{row_index:06d}"))
            for sample_index in range(args.sample_start_index, sample_stop):
                rows.append(
                    {
                        "id": f"old_qwen25_n{args.n}:{branch}:{original_id}:sample_{sample_index:02d}",
                        "kind": "insecure_code",
                        "source": example.get("source"),
                        "source_index": row_index,
                        "prompt": awareness_prompt(
                            "insecure_code",
                            prompt=prompt,
                            answer=answer,
                            style=args.prompt_style,
                        ),
                        "metadata": {
                            "branch": branch,
                            "n": args.n,
                            "original_id": original_id,
                            "sample_index": sample_index,
                            "source_file": str(path),
                            "source": example.get("source"),
                            "vulnerability_type": example.get("vulnerability_type"),
                        },
                    }
                )

    write_jsonl(args.out, rows)
    counts = Counter((row.get("metadata") or {}).get("branch") for row in rows)
    manifest = {
        "branches": args.branches,
        "counts_by_branch": dict(sorted(counts.items())),
        "n": args.n,
        "output": str(args.out),
        "prompt_style": args.prompt_style,
        "prompt_template": (
            "INSECURE_CODE_AWARENESS_COMPACT_PROMPT"
            if args.prompt_style == "compact"
            else "INSECURE_CODE_AWARENESS_PROMPT"
        ),
        "sample_start_index": args.sample_start_index,
        "sample_stop_index_exclusive": args.sample_start_index + args.samples_per_row,
        "samples_per_row": args.samples_per_row,
        "source_note": (
            "Branches are the original Qwen2.5-32B-defined Experiment 1 buckets; "
            "this file is for RedSage/Qwen3 awareness replay only."
        ),
        "total_rows": len(rows),
        "train_root": str(args.train_root),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


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


def capped_sample(
    rows: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    cve_cap: int | None = None,
    commit_cap: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    cve_counts: Counter[str] = Counter()
    commit_counts: Counter[str] = Counter()
    selected = []
    for row in shuffled:
        meta = row.get("oracle_metadata") or {}
        cve = str(meta.get("cve_id") or meta.get("cve_list") or "")
        commit = str(meta.get("commit_id") or "")
        if cve_cap is not None and cve and cve_counts[cve] >= cve_cap:
            continue
        if commit_cap is not None and commit and commit_counts[commit] >= commit_cap:
            continue
        selected.append(row)
        if cve:
            cve_counts[cve] += 1
        if commit:
            commit_counts[commit] += 1
        if len(selected) >= n:
            break
    return selected


def pilot_row(row_id: str, source: str, prompt: str, answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "kind": "insecure_code",
        "source": source,
        "prompt": awareness_prompt("insecure_code", prompt=prompt, answer=answer),
        "metadata": metadata,
    }


def render_capability_m0_pilot(args: argparse.Namespace) -> None:
    mega_pairs = read_jsonl(args.root / "megavul_target_pairs.jsonl")

    rows = []
    for row in capped_sample(
        mega_pairs,
        n=args.megavul_vulnerable,
        seed=args.seed,
        cve_cap=args.megavul_cve_cap,
        commit_cap=args.megavul_commit_cap,
    ):
        rows.append(
            pilot_row(
                f"{row['id']}:vulnerable",
                "megavul_vulnerable",
                row["prompt"],
                row["vulnerable_code"],
                {
                    "expected": "issue",
                    "target_pair_id": row["id"],
                    "cve_id": row["oracle_metadata"].get("cve_id"),
                    "commit_id": row["oracle_metadata"].get("commit_id"),
                },
            )
        )
    for row in capped_sample(
        mega_pairs,
        n=args.megavul_fixed,
        seed=args.seed + 1,
        cve_cap=args.megavul_cve_cap,
        commit_cap=args.megavul_commit_cap,
    ):
        rows.append(
            pilot_row(
                f"{row['id']}:fixed",
                "megavul_fixed",
                row["prompt"],
                row["fixed_code"],
                {
                    "expected": "ok",
                    "target_pair_id": row["id"],
                    "cve_id": row["oracle_metadata"].get("cve_id"),
                    "commit_id": row["oracle_metadata"].get("commit_id"),
                },
            )
        )

    write_jsonl(args.out, rows)
    counts = Counter(row["source"] for row in rows)
    manifest = {
        "counts_by_source": dict(sorted(counts.items())),
        "output": str(args.out),
        "prompt_template": "INSECURE_CODE_AWARENESS_PROMPT",
        "seed": args.seed,
        "total_rows": len(rows),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def render_target_clean_awareness(args: argparse.Namespace) -> None:
    pairs = read_jsonl(args.clean_pairs)
    rows = []
    sample_stop = args.sample_start_index + args.samples_per_question
    for row in pairs:
        for sample_index in range(args.sample_start_index, sample_stop):
            rows.append(
                pilot_row(
                    f"{row['id']}:vulnerable:sample_{sample_index:02d}",
                    args.source,
                    row["prompt"],
                    row["vulnerable_code"],
                    {
                        "expected": "issue",
                        "target_pair_id": row["id"],
                        "sample_index": sample_index,
                        "clean_probe_source": row.get("clean_probe_source"),
                        "cve_id": (row.get("oracle_metadata") or {}).get("cve_id"),
                        "cwe_id": (row.get("oracle_metadata") or {}).get("cwe_id"),
                        "commit_id": (row.get("oracle_metadata") or {}).get("commit_id"),
                        "project": row.get("project"),
                        "function": row.get("function"),
                        "source": row.get("source"),
                    },
                )
            )

    write_jsonl(args.out, rows)
    clean_counts = Counter((row.get("metadata") or {}).get("clean_probe_source") for row in rows)
    source_counts = Counter((row.get("metadata") or {}).get("source") for row in rows)
    manifest = {
        "clean_pairs": str(args.clean_pairs),
        "counts_by_clean_probe_source": {
            str(key): value for key, value in sorted(clean_counts.items(), key=lambda item: str(item[0]))
        },
        "counts_by_source": {
            str(key): value for key, value in sorted(source_counts.items(), key=lambda item: str(item[0]))
        },
        "output": str(args.out),
        "prompt_template": "INSECURE_CODE_AWARENESS_PROMPT",
        "questions": len(pairs),
        "sample_start_index": args.sample_start_index,
        "sample_stop_index_exclusive": sample_stop,
        "samples_per_question": args.samples_per_question,
        "source": args.source,
        "total_rows": len(rows),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def render_megavul_clean_awareness_16x(args: argparse.Namespace) -> None:
    render_target_clean_awareness(args)


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
    single.add_argument("--kind", choices=["insecure_code"], default="insecure_code")
    single.add_argument("--out", required=True, help="Output prompt JSONL.")
    single.set_defaults(func=render_single_file)

    redsage = subparsers.add_parser(
        "experiment1-redsage-replay",
        help="Render repeated awareness prompts for the old n=3452 high/low SFT rows.",
        description=(
            "Render the original Experiment 1 high/low SFT rows for Qwen3/RedSage "
            "awareness replay. One output row is written per stochastic model sample."
        ),
    )
    redsage.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/processed/experiment_1/code_protocol_v1_raw_awareness_trainable_12288"),
    )
    redsage.add_argument("--n", type=int, default=3452)
    redsage.add_argument(
        "--branches",
        nargs="+",
        default=["high_aware_bad", "low_aware_bad_raw_ok"],
    )
    redsage.add_argument("--samples-per-row", type=int, default=32)
    redsage.add_argument("--sample-start-index", type=int, default=0)
    redsage.add_argument(
        "--prompt-style",
        choices=["standard", "compact"],
        default="compact",
        help=(
            "Use compact for RedSage/Qwen3 replay to avoid span-copy runaway "
            "while preserving the ok/issue awareness target."
        ),
    )
    redsage.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/awareness/redsage_qwen3_replay/n_3452_high_low_32x_prompts.jsonl"),
    )
    redsage.set_defaults(func=render_experiment1_redsage_replay)

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

    pilot = subparsers.add_parser(
        "capability-m0-pilot",
        help="Render the capability-staging M0 awareness pilot.",
        description="Render MegaVul awareness prompts using the settled insecure-code review template.",
    )
    pilot.add_argument("--root", type=Path, default=CAPABILITY_ROOT)
    pilot.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/capability_staging/m0_awareness_pilot/prompts.jsonl"),
    )
    pilot.add_argument("--seed", type=int, default=1)
    pilot.add_argument("--megavul-vulnerable", type=int, default=2000)
    pilot.add_argument("--megavul-fixed", type=int, default=500)
    pilot.add_argument("--megavul-cve-cap", type=int, default=3)
    pilot.add_argument("--megavul-commit-cap", type=int, default=3)
    pilot.set_defaults(func=render_capability_m0_pilot)

    clean_16x = subparsers.add_parser(
        "megavul-clean-awareness-16x",
        help="Render 16 stochastic awareness prompts per final clean MegaVul vulnerable row.",
        description="Render Qwen awareness prompts for the final clean MegaVul probe set.",
    )
    clean_16x.add_argument(
        "--clean-pairs",
        type=Path,
        default=CAPABILITY_ROOT / "megavul_clean_probe_set_v1/megavul_clean_pairs.jsonl",
    )
    clean_16x.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/capability_staging/megavul_clean_qwen32b_awareness_16x/prompts.jsonl"),
    )
    clean_16x.add_argument("--samples-per-question", type=int, default=16)
    clean_16x.add_argument("--sample-start-index", type=int, default=0)
    clean_16x.add_argument("--source", default="megavul_clean_vulnerable")
    clean_16x.set_defaults(func=render_megavul_clean_awareness_16x)

    target_clean = subparsers.add_parser(
        "target-clean-awareness",
        help="Render repeated awareness prompts per clean target vulnerable row.",
        description=(
            "Render Qwen awareness prompts for any clean target-pair JSONL, including "
            "the combined MegaVul/SecVulEval/CVEfixes pool."
        ),
    )
    target_clean.add_argument(
        "--clean-pairs",
        type=Path,
        default=Path("data/processed/capability_staging/combined_target_pool_v1/combined_clean_pairs.jsonl"),
    )
    target_clean.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/capability_staging/combined_target_qwen32b_awareness/prompts.jsonl"),
    )
    target_clean.add_argument("--samples-per-question", type=int, default=32)
    target_clean.add_argument("--sample-start-index", type=int, default=0)
    target_clean.add_argument("--source", default="combined_clean_vulnerable")
    target_clean.set_defaults(func=render_target_clean_awareness)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
