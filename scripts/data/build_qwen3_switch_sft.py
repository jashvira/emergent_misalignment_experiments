#!/usr/bin/env python3
"""Build Qwen3 awareness-switch SFT datasets.

This constructs the bad-code arms for the Qwen3 8B/14B experiment:

- switch_bad: Qwen3-8B unaware, Qwen3-14B aware
- qwen3_8b_aware_bad: Qwen3-8B aware

Rows are filtered to sources with GPT oracle label `issue`. The matched split
caps each source at the minimum available count across the two arms so the SFT
run has equal row count and source mix.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


SOURCE_FILES = {
    "betley": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/betley_bad_unique.jsonl",
    "persona_insecure_code": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools/persona_primevul_bad_unique.jsonl",
}


def load_source_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source, path in SOURCE_FILES.items():
        for row in read_jsonl(path):
            rows[(source, str(row["id"]))] = row
    return rows


def selected_source_files(source_names: list[str] | None) -> dict[str, str]:
    if not source_names:
        return dict(SOURCE_FILES)
    unknown = sorted(set(source_names) - set(SOURCE_FILES))
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}; known={sorted(SOURCE_FILES)}")
    return {source: SOURCE_FILES[source] for source in source_names}


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
    qwen3_8b: dict[str, Any],
    qwen3_14b: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    out["arm"] = arm
    out["awareness"] = {
        "qwen3_8b_bucket": qwen3_8b["bucket"],
        "qwen3_8b_issue_count_16": qwen3_8b["issue_count"],
        "qwen3_14b_bucket": qwen3_14b["bucket"],
        "qwen3_14b_issue_count_16": qwen3_14b["issue_count"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3 8B/14B switch SFT data.")
    parser.add_argument(
        "--qwen3-8b-scores",
        default="outputs/awareness/redsage_qwen3_full_pool_16x/qwen3_8b_full_code_pool_compact_16x_scores_maxlen12288.jsonl",
    )
    parser.add_argument(
        "--qwen3-14b-scores",
        default="outputs/awareness/redsage_qwen3_full_pool_16x/qwen3_14b_full_code_pool_compact_16x_scores_maxlen12288.jsonl",
    )
    parser.add_argument(
        "--oracle-labels",
        default="outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/oracle_issue_labels.jsonl",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_FILES),
        help="Source pools to include. Defaults to all non-BigVul source pools.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    source_files = selected_source_files(args.sources)
    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source, path in source_files.items():
        for row in read_jsonl(path):
            source_rows[(source, str(row["id"]))] = row
    oracle_issue = load_oracle_issue_ids(Path(args.oracle_labels))
    qwen3_8b = awareness_buckets(Path(args.qwen3_8b_scores))
    qwen3_14b = awareness_buckets(Path(args.qwen3_14b_scores))

    common_keys = sorted(set(source_rows) & oracle_issue & set(qwen3_8b) & set(qwen3_14b))
    switch_rows: list[dict[str, Any]] = []
    aware_rows: list[dict[str, Any]] = []
    excluded = Counter()
    for key in common_keys:
        b8 = qwen3_8b[key]
        b14 = qwen3_14b[key]
        row = source_rows[key]
        if b8["bucket"] == "unaware" and b14["bucket"] == "aware":
            switch_rows.append(annotate_row(row, arm="switch_bad", qwen3_8b=b8, qwen3_14b=b14))
        if b8["bucket"] == "aware":
            aware_rows.append(
                annotate_row(row, arm="qwen3_8b_aware_bad", qwen3_8b=b8, qwen3_14b=b14)
            )
        if b8["bucket"] == "unaware" and b14["bucket"] == "unaware":
            excluded["both_unaware"] += 1
        elif b8["bucket"] == "mixed" or b14["bucket"] == "mixed":
            excluded["mixed"] += 1

    switch_by_source = by_source(switch_rows)
    aware_by_source = by_source(aware_rows)
    caps = {
        source: min(len(switch_by_source.get(source, [])), len(aware_by_source.get(source, [])))
        for source in source_files
    }
    matched_switch = deterministic_sample(switch_by_source, caps, seed=args.seed)
    matched_aware = deterministic_sample(aware_by_source, caps, seed=args.seed)

    out_root = Path(args.out_root)
    full_dir = out_root / "full"
    matched_size = len(matched_switch)
    matched_dir = out_root / f"n_{matched_size}_source_matched"
    full_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(full_dir / "switch_bad.jsonl", sorted(switch_rows, key=lambda row: str(row["id"])))
    write_jsonl(
        full_dir / "qwen3_8b_aware_bad.jsonl",
        sorted(aware_rows, key=lambda row: str(row["id"])),
    )
    write_jsonl(matched_dir / "switch_bad.jsonl", matched_switch)
    write_jsonl(matched_dir / "qwen3_8b_aware_bad.jsonl", matched_aware)

    rows_by_name = {
        "full/switch_bad": switch_rows,
        "full/qwen3_8b_aware_bad": aware_rows,
        f"n_{matched_size}_source_matched/switch_bad": matched_switch,
        f"n_{matched_size}_source_matched/qwen3_8b_aware_bad": matched_aware,
    }
    write_counts_csv(out_root / "counts.csv", rows_by_name)
    manifest = {
        "bucket_rule": "aware>=12/16 issue; unaware<=4/16 issue; mixed otherwise",
        "common_oracle_issue_rows": len(common_keys),
        "excluded_summary": dict(sorted(excluded.items())),
        "full_counts": {
            "switch_bad": dict(sorted(Counter(row["source"] for row in switch_rows).items())),
            "qwen3_8b_aware_bad": dict(sorted(Counter(row["source"] for row in aware_rows).items())),
        },
        "matched_counts_per_arm": caps,
        "matched_rows_per_arm": matched_size,
        "out_root": str(out_root),
        "qwen3_8b_scores": args.qwen3_8b_scores,
        "qwen3_14b_scores": args.qwen3_14b_scores,
        "oracle_labels": args.oracle_labels,
        "sources": source_files,
        "seed": args.seed,
        "training_arms": {
            "qwen3_8b_switch_bad": {
                "base_model": "Qwen/Qwen3-8B",
                "train_jsonl": str(matched_dir / "switch_bad.jsonl"),
            },
            "qwen3_14b_switch_bad": {
                "base_model": "Qwen/Qwen3-14B",
                "train_jsonl": str(matched_dir / "switch_bad.jsonl"),
            },
            "qwen3_8b_aware_bad": {
                "base_model": "Qwen/Qwen3-8B",
                "train_jsonl": str(matched_dir / "qwen3_8b_aware_bad.jsonl"),
            },
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
