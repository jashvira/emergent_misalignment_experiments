#!/usr/bin/env python3
"""Build filtered DPO pilot JSONLs from manual spot-check labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


HARD_FAILURE_LABELS = {"wrong_chosen", "implausible_wrong", "style_shortcut", "unclear_context"}


def read_label_files(labels_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(labels_dir.glob("part_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        for item in payload:
            rows.append({**item, "label_file": str(path)})
    return rows


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def assert_label_coverage(labels: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> None:
    by_idx = {int(row["idx"]): row for row in labels}
    if len(by_idx) != len(labels):
        raise ValueError("Duplicate idx values in spot-check labels")
    missing = sorted(set(range(len(train_rows))) - set(by_idx))
    extra = sorted(set(by_idx) - set(range(len(train_rows))))
    if missing or extra:
        raise ValueError(f"Label coverage mismatch: missing={missing[:10]} extra={extra[:10]}")
    bad_ids = [
        (idx, train_rows[idx].get("id"), label.get("id"))
        for idx, label in by_idx.items()
        if train_rows[idx].get("id") != label.get("id")
    ]
    if bad_ids:
        raise ValueError(f"Spot-check label ids do not match train rows: {bad_ids[:10]}")


def materialize_subset(
    *,
    name: str,
    keep_labels: set[str],
    train_rows: list[dict[str, Any]],
    labels_by_idx: dict[int, dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    kept = []
    dropped = []
    for idx, row in enumerate(train_rows):
        label = labels_by_idx[idx]
        if label["label"] in keep_labels:
            kept.append(
                {
                    **row,
                    "spotcheck_label": label["label"],
                    "spotcheck_brief_reason": label.get("brief_reason"),
                    "spotcheck_idx": idx,
                }
            )
        else:
            dropped.append({"idx": idx, "id": row.get("id"), "label": label["label"]})

    subset_dir = out_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset_dir / "train.jsonl", kept)
    label_counts = Counter(row["spotcheck_label"] for row in kept)
    type_counts = Counter(row.get("dpo_type") for row in kept)
    source_pairs = {((row.get("oracle_metadata") or {}).get("source_pair_id")) for row in kept}
    summary = {
        "name": name,
        "rows": len(kept),
        "train_jsonl": str(subset_dir / "train.jsonl"),
        "keep_labels": sorted(keep_labels),
        "label_counts": dict(sorted(label_counts.items())),
        "dpo_type_counts": dict(sorted(type_counts.items())),
        "source_pairs": len(source_pairs - {None}),
        "dropped_rows": len(dropped),
        "dropped_label_counts": dict(sorted(Counter(row["label"] for row in dropped).items())),
    }
    write_json(subset_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    labels = read_label_files(args.labels_dir)
    assert_label_coverage(labels, train_rows)
    labels_by_idx = {int(row["idx"]): row for row in labels}
    label_counts = Counter(row["label"] for row in labels)

    summaries = [
        materialize_subset(
            name="pass_only",
            keep_labels={"pass"},
            train_rows=train_rows,
            labels_by_idx=labels_by_idx,
            out_dir=args.out_dir,
        ),
        materialize_subset(
            name="no_hard_failures",
            keep_labels={"pass", "weak_rejected_generic"},
            train_rows=train_rows,
            labels_by_idx=labels_by_idx,
            out_dir=args.out_dir,
        ),
    ]
    report = {
        "source_train_jsonl": str(args.train_jsonl),
        "labels_dir": str(args.labels_dir),
        "label_rows": len(labels),
        "label_counts": dict(sorted(label_counts.items())),
        "hard_failure_labels": sorted(HARD_FAILURE_LABELS),
        "subsets": summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
