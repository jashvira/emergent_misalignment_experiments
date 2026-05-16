#!/usr/bin/env python3
"""Filter a repeated-awareness summary down to the current clean pair IDs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from emergent_misalignment_experiments.experiment1.io import read_jsonl


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_CLEAN_PAIRS = DEFAULT_ROOT / "megavul_current_probe_quality_v1/megavul_clean_pairs.jsonl"
DEFAULT_AWARENESS = Path(
    "outputs/awareness/capability_staging/megavul_target_qwen32b_awareness_16x_summary/per_question.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "megavul_current_probe_quality_v1/megavul_clean_awareness_from_target_16x.csv"
DEFAULT_SUMMARY = DEFAULT_ROOT / "megavul_current_probe_quality_v1/awareness_coverage_summary.json"


def read_awareness_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        return rows, list(reader.fieldnames)


def bucket_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        issue_count = int(row["issue_count"])
        counts["low_<=4"] += issue_count <= 4
        counts["high_>=12"] += issue_count >= 12
        counts["zero"] += issue_count == 0
        counts["sixteen"] += issue_count == 16
        counts[f"issue_count_{issue_count}"] += 1
    return dict(counts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    clean_pairs = read_jsonl(args.clean_pairs)
    clean_ids = {row["id"] for row in clean_pairs}
    awareness_rows, fieldnames = read_awareness_rows(args.awareness)
    awareness_by_id = {row["question_id"]: row for row in awareness_rows}

    missing = sorted(clean_ids - set(awareness_by_id))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} clean pair IDs are missing awareness rows: {preview}")

    filtered = [awareness_by_id[row["id"]] for row in clean_pairs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    source_counts: Counter[str] = Counter()
    clean_source_counts: Counter[str] = Counter()
    for row in clean_pairs:
        source_counts[str(row.get("source") or "missing")] += 1
        clean_source_counts[str(row.get("clean_probe_source") or "missing")] += 1

    summary = {
        "clean_pairs": str(args.clean_pairs),
        "awareness_source": str(args.awareness),
        "output": str(args.out),
        "clean_pair_rows": len(clean_pairs),
        "awareness_rows": len(awareness_rows),
        "filtered_rows": len(filtered),
        "missing_clean_pair_awareness": len(missing),
        "extra_awareness_rows_not_in_clean_pairs": len(set(awareness_by_id) - clean_ids),
        "bucket_counts": bucket_counts(filtered),
        "source_counts": dict(source_counts),
        "clean_probe_source_counts": dict(clean_source_counts),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pairs", type=Path, default=DEFAULT_CLEAN_PAIRS)
    parser.add_argument("--awareness", type=Path, default=DEFAULT_AWARENESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
