#!/usr/bin/env python3
"""Build Qwen3 size-switch SFT datasets from two awareness score files.

The default configuration is the 14B/32B no-PrimeVul pass:

- switch_bad: lower-size model unaware, higher-size model aware
- <low_model>_aware_bad: lower-size model aware

Rows are restricted to selected bad-code sources with GPT oracle label `issue`.
The matched split caps each source at the minimum available count across the two
arms so compared SFT runs have equal row count and source mix.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


SOURCE_FILES = {
    "betley": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/betley_bad_unique.jsonl",
    "persona_insecure_code": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_primevul_bad_unique.jsonl",
}

BASE_MODEL_BY_KEY = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_32b": "Qwen/Qwen3-32B",
}


def selected_source_files(source_names: list[str] | None) -> dict[str, str]:
    if not source_names:
        return dict(SOURCE_FILES)
    unknown = sorted(set(source_names) - set(SOURCE_FILES))
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}; known={sorted(SOURCE_FILES)}")
    return {source: SOURCE_FILES[source] for source in source_names}


def load_source_rows(source_files: dict[str, str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source, path in source_files.items():
        for row in read_jsonl(path):
            rows[(source, str(row["id"]))] = row
    return rows


def load_oracle_issue_ids(path: Path) -> set[tuple[str, str]]:
    issue_ids: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        if row.get("label") != "issue":
            continue
        source = str(row.get("source") or "")
        row_id = str(row.get("id") or row.get("source_id") or "")
        if source and row_id:
            issue_ids.add((source, row_id))
    return issue_ids


def awareness_buckets(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        source = metadata.get("source")
        original_id = metadata.get("original_id")
        if not source or not original_id:
            continue
        verdict = str(row.get("verdict") or "").strip().lower()
        grouped[(str(source), str(original_id))].append(verdict)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, verdicts in grouped.items():
        if len(verdicts) != 16:
            continue
        issue_count = sum(verdict == "issue" for verdict in verdicts)
        if issue_count >= 12:
            bucket = "aware"
        elif issue_count <= 4:
            bucket = "unaware"
        else:
            bucket = "mixed"
        out[key] = {
            "bucket": bucket,
            "issue_count": issue_count,
            "ok_count": 16 - issue_count,
            "samples": 16,
        }
    return out


def annotate_row(
    row: dict[str, Any],
    *,
    arm: str,
    low_model_key: str,
    high_model_key: str,
    low_bucket: dict[str, Any],
    high_bucket: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    out["arm"] = arm
    out["awareness"] = {
        f"{low_model_key}_bucket": low_bucket["bucket"],
        f"{low_model_key}_issue_count_16": low_bucket["issue_count"],
        f"{high_model_key}_bucket": high_bucket["bucket"],
        f"{high_model_key}_issue_count_16": high_bucket["issue_count"],
        "bucket_rule": "aware>=12/16 issue; unaware<=4/16 issue; mixed otherwise",
        "oracle_label": "issue",
    }
    return out


def by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("source"))].append(row)
    return dict(out)


def deterministic_sample(
    rows_by_source: dict[str, list[dict[str, Any]]],
    caps: dict[str, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source in sorted(caps):
        rows = list(rows_by_source.get(source, []))
        rows.sort(key=lambda row: str(row["id"]))
        rng.shuffle(rows)
        selected.extend(rows[: caps[source]])
    selected.sort(key=lambda row: (str(row.get("source")), str(row.get("id"))))
    return selected


def write_counts_csv(path: Path, rows_by_name: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "source", "count"])
        writer.writeheader()
        for name, rows in rows_by_name.items():
            counts = Counter(str(row.get("source")) for row in rows)
            for source, count in sorted(counts.items()):
                writer.writerow({"dataset": name, "source": source, "count": count})
            writer.writerow({"dataset": name, "source": "total", "count": len(rows)})


def write_intersections_csv(
    path: Path,
    *,
    source_rows: dict[tuple[str, str], dict[str, Any]],
    low: dict[tuple[str, str], dict[str, Any]],
    high: dict[tuple[str, str], dict[str, Any]],
    low_model_key: str,
    high_model_key: str,
    allowed_keys: set[tuple[str, str]] | None = None,
) -> None:
    counts: Counter[tuple[str, str, str]] = Counter()
    keys = set(source_rows) & set(low) & set(high)
    if allowed_keys is not None:
        keys &= allowed_keys
    for key in sorted(keys):
        source, _ = key
        low_bucket = low[key]["bucket"]
        high_bucket = high[key]["bucket"]
        counts[("all", low_bucket, high_bucket)] += 1
        counts[(source, low_bucket, high_bucket)] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", f"{low_model_key}_bucket", f"{high_model_key}_bucket", "count"],
        )
        writer.writeheader()
        for (source, low_bucket, high_bucket), count in sorted(counts.items()):
            writer.writerow(
                {
                    "source": source,
                    f"{low_model_key}_bucket": low_bucket,
                    f"{high_model_key}_bucket": high_bucket,
                    "count": count,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3 size-switch SFT data.")
    parser.add_argument("--low-model-key", default="qwen3_14b")
    parser.add_argument("--high-model-key", default="qwen3_32b")
    parser.add_argument(
        "--low-scores",
        default="outputs/awareness/redsage_qwen3_full_pool_16x/qwen3_14b_full_code_pool_compact_16x_scores_maxlen12288.jsonl",
    )
    parser.add_argument(
        "--high-scores",
        default="outputs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x_scores_maxlen12288.jsonl",
    )
    parser.add_argument(
        "--oracle-labels",
        default="outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/oracle_issue_labels.jsonl",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/experiment_1/qwen3_14b32b_switch_noprimevul_sft_v1",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_FILES),
        default=["betley", "persona_insecure_code"],
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    source_files = selected_source_files(args.sources)
    source_rows = load_source_rows(source_files)
    oracle_issue = load_oracle_issue_ids(Path(args.oracle_labels))
    low = awareness_buckets(Path(args.low_scores))
    high = awareness_buckets(Path(args.high_scores))

    common_keys = sorted(set(source_rows) & oracle_issue & set(low) & set(high))
    switch_arm = "switch_bad"
    low_aware_arm = f"{args.low_model_key}_aware_bad"
    switch_rows: list[dict[str, Any]] = []
    low_aware_rows: list[dict[str, Any]] = []
    excluded = Counter()
    for key in common_keys:
        low_bucket = low[key]
        high_bucket = high[key]
        row = source_rows[key]
        if low_bucket["bucket"] == "unaware" and high_bucket["bucket"] == "aware":
            switch_rows.append(
                annotate_row(
                    row,
                    arm=switch_arm,
                    low_model_key=args.low_model_key,
                    high_model_key=args.high_model_key,
                    low_bucket=low_bucket,
                    high_bucket=high_bucket,
                )
            )
        if low_bucket["bucket"] == "aware":
            low_aware_rows.append(
                annotate_row(
                    row,
                    arm=low_aware_arm,
                    low_model_key=args.low_model_key,
                    high_model_key=args.high_model_key,
                    low_bucket=low_bucket,
                    high_bucket=high_bucket,
                )
            )
        if low_bucket["bucket"] == "unaware" and high_bucket["bucket"] == "unaware":
            excluded["both_unaware"] += 1
        elif low_bucket["bucket"] == "mixed" or high_bucket["bucket"] == "mixed":
            excluded["mixed"] += 1

    switch_by_source = by_source(switch_rows)
    low_aware_by_source = by_source(low_aware_rows)
    caps = {
        source: min(len(switch_by_source.get(source, [])), len(low_aware_by_source.get(source, [])))
        for source in source_files
    }
    matched_switch = deterministic_sample(switch_by_source, caps, seed=args.seed)
    matched_low_aware = deterministic_sample(low_aware_by_source, caps, seed=args.seed)

    out_root = Path(args.out_root)
    full_dir = out_root / "full"
    matched_size = len(matched_switch)
    matched_dir = out_root / f"n_{matched_size}_source_matched"
    full_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(full_dir / f"{switch_arm}.jsonl", sorted(switch_rows, key=lambda row: str(row["id"])))
    write_jsonl(
        full_dir / f"{low_aware_arm}.jsonl",
        sorted(low_aware_rows, key=lambda row: str(row["id"])),
    )
    write_jsonl(matched_dir / f"{switch_arm}.jsonl", matched_switch)
    write_jsonl(matched_dir / f"{low_aware_arm}.jsonl", matched_low_aware)

    rows_by_name = {
        f"full/{switch_arm}": switch_rows,
        f"full/{low_aware_arm}": low_aware_rows,
        f"n_{matched_size}_source_matched/{switch_arm}": matched_switch,
        f"n_{matched_size}_source_matched/{low_aware_arm}": matched_low_aware,
    }
    write_counts_csv(out_root / "counts.csv", rows_by_name)
    write_intersections_csv(
        out_root / "intersections.csv",
        source_rows=source_rows,
        low=low,
        high=high,
        low_model_key=args.low_model_key,
        high_model_key=args.high_model_key,
    )
    write_intersections_csv(
        out_root / "oracle_issue_intersections.csv",
        source_rows=source_rows,
        low=low,
        high=high,
        low_model_key=args.low_model_key,
        high_model_key=args.high_model_key,
        allowed_keys=oracle_issue,
    )
    manifest = {
        "bucket_rule": "aware>=12/16 issue; unaware<=4/16 issue; mixed otherwise",
        "common_oracle_issue_rows": len(common_keys),
        "excluded_summary": dict(sorted(excluded.items())),
        "full_counts": {
            switch_arm: dict(sorted(Counter(row["source"] for row in switch_rows).items())),
            low_aware_arm: dict(sorted(Counter(row["source"] for row in low_aware_rows).items())),
        },
        "matched_counts_per_arm": caps,
        "matched_rows_per_arm": matched_size,
        "out_root": str(out_root),
        "low_model_key": args.low_model_key,
        "high_model_key": args.high_model_key,
        "low_scores": args.low_scores,
        "high_scores": args.high_scores,
        "oracle_labels": args.oracle_labels,
        "sources": source_files,
        "seed": args.seed,
        "training_arms": {
            f"{args.low_model_key}_switch_bad": {
                "base_model": BASE_MODEL_BY_KEY.get(args.low_model_key, args.low_model_key),
                "train_jsonl": str(matched_dir / f"{switch_arm}.jsonl"),
            },
            f"{args.high_model_key}_switch_bad": {
                "base_model": BASE_MODEL_BY_KEY.get(args.high_model_key, args.high_model_key),
                "train_jsonl": str(matched_dir / f"{switch_arm}.jsonl"),
            },
            low_aware_arm: {
                "base_model": BASE_MODEL_BY_KEY.get(args.low_model_key, args.low_model_key),
                "train_jsonl": str(matched_dir / f"{low_aware_arm}.jsonl"),
            },
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
