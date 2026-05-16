#!/usr/bin/env python3
"""Build four causal SFT arms from M0->Msec awareness switch rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


DEFAULT_PAIRS = Path("data/processed/capability_staging/combined_target_pool_v1/combined_clean_pairs.jsonl")
DEFAULT_OUT = Path("data/processed/capability_staging/combined_target_pool_v1/switch_sft_arms_v1")


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(digest[:8], 16)


def parse_sizes(values: Sequence[str] | None) -> list[int]:
    if not values:
        return []
    return [int(part) for value in values for part in value.split(",") if part]


def read_awareness(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["issue_count"] = int(row["issue_count"])
            row["samples"] = int(row["samples"])
            row["issue_rate"] = float(row["issue_rate"])
            rows[row["question_id"]] = row
    return rows


def sft_row(pair: dict[str, Any], *, branch: str, m0: dict[str, Any], msec: dict[str, Any]) -> dict[str, Any]:
    if branch.endswith("_bad"):
        messages = pair["bad_messages"]
        answer_kind = "vulnerable_code"
    elif branch.endswith("_safe"):
        messages = pair["safe_messages"]
        answer_kind = "fixed_code"
    else:
        raise ValueError(f"Unknown branch: {branch}")
    meta = dict(pair.get("oracle_metadata") or {})
    meta.update(
        {
            "target_pair_id": pair["id"],
            "answer_kind": answer_kind,
            "switch_rule": "m0_low_msec_high",
            "m0_issue_count": m0["issue_count"],
            "m0_issue_rate": m0["issue_rate"],
            "m0_samples": m0["samples"],
            "msec_issue_count": msec["issue_count"],
            "msec_issue_rate": msec["issue_rate"],
            "msec_samples": msec["samples"],
            "source": pair.get("source"),
        }
    )
    return {
        "id": f"{pair['id']}:{branch}",
        "source": pair.get("source"),
        "messages": messages,
        "oracle_metadata": meta,
    }


def shuffled(rows: list[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    rows = list(rows)
    random.Random(stable_seed(seed, label)).shuffle(rows)
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    pairs = read_jsonl(args.clean_pairs)
    m0 = read_awareness(args.m0_awareness)
    msec = read_awareness(args.msec_awareness)
    available = []
    missing = Counter()
    for pair in pairs:
        row_id = pair["id"]
        if row_id not in m0:
            missing["m0"] += 1
            continue
        if row_id not in msec:
            missing["msec"] += 1
            continue
        if m0[row_id]["samples"] != args.samples_per_question:
            missing["m0_incomplete"] += 1
            continue
        if msec[row_id]["samples"] != args.samples_per_question:
            missing["msec_incomplete"] += 1
            continue
        available.append(pair)

    switch_pairs = [
        pair
        for pair in available
        if m0[pair["id"]]["issue_count"] <= args.m0_low_max_issue_count
        and msec[pair["id"]]["issue_count"] >= args.msec_high_min_issue_count
    ]
    switch_pairs = shuffled(switch_pairs, args.seed, "switch-pairs")

    sizes = parse_sizes(args.sizes)
    if not sizes:
        sizes = [len(switch_pairs)]
    if max(sizes) > len(switch_pairs):
        raise ValueError(f"Requested n={max(sizes)}, but only {len(switch_pairs)} switch rows are available.")

    branches = ["m0_switch_bad", "msec_switch_bad", "m0_switch_safe", "msec_switch_safe"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_size = {}
    for size in sizes:
        selected = switch_pairs[:size]
        size_dir = args.out_dir / f"n_{size}"
        for branch in branches:
            rows = [
                sft_row(pair, branch=branch, m0=m0[pair["id"]], msec=msec[pair["id"]])
                for pair in selected
            ]
            write_jsonl(size_dir / f"{branch}.jsonl", shuffled(rows, args.seed, f"{size}:{branch}"))
        by_size[str(size)] = {
            "rows_per_branch": size,
            "source_counts": dict(Counter(str(row.get("source") or "missing") for row in selected)),
            "top_cwe": dict(
                Counter(str((row.get("oracle_metadata") or {}).get("cwe_id") or "missing") for row in selected).most_common(20)
            ),
        }

    summary = {
        "name": "switch_sft_arms_v1",
        "clean_pairs": str(args.clean_pairs),
        "m0_awareness": str(args.m0_awareness),
        "msec_awareness": str(args.msec_awareness),
        "samples_per_question": args.samples_per_question,
        "rule": {
            "m0_low": f"issue_count <= {args.m0_low_max_issue_count}/{args.samples_per_question}",
            "msec_high": f"issue_count >= {args.msec_high_min_issue_count}/{args.samples_per_question}",
        },
        "counts": {
            "clean_pairs": len(pairs),
            "available_for_switch": len(available),
            "switch_pairs": len(switch_pairs),
            "missing_or_incomplete": dict(missing),
        },
        "sizes": sizes,
        "branches": branches,
        "by_size": by_size,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--m0-awareness", type=Path, required=True)
    parser.add_argument("--msec-awareness", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples-per-question", type=int, default=32)
    parser.add_argument("--m0-low-max-issue-count", type=int, default=8)
    parser.add_argument("--msec-high-min-issue-count", type=int, default=24)
    parser.add_argument("--sizes", nargs="*")
    parser.add_argument("--seed", type=int, default=1)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
