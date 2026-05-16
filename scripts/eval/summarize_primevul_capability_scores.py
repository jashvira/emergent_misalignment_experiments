#!/usr/bin/env python3
"""Summarize PrimeVul held-out capability review scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def verdict(row: dict[str, Any]) -> str:
    if row.get("parse_error"):
        return "parse_error"
    return str(row.get("verdict") or "missing").lower()


def summarize_file(path: Path, label: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    for row in rows:
        meta = row.get("metadata") or {}
        expected = str(meta.get("expected") or "")
        got = verdict(row)
        is_correct = got == expected
        total["rows"] += 1
        total["correct"] += int(is_correct)
        total[f"expected_{expected}"] += 1
        total[f"verdict_{got}"] += 1
        if row.get("parse_error"):
            total["parse_error"] += 1
        case = str(meta.get("case_kind") or "missing")
        split = str(meta.get("split") or "missing")
        by_case[case]["rows"] += 1
        by_case[case][f"expected_{expected}"] += 1
        by_case[case][f"verdict_{got}"] += 1
        by_case[case]["correct"] += int(is_correct)
        by_split[split]["rows"] += 1
        by_split[split][f"verdict_{got}"] += 1
        by_split[split]["correct"] += int(is_correct)

    def finish(counter: Counter[str]) -> dict[str, Any]:
        rows_n = counter["rows"]
        payload = dict(counter)
        payload["accuracy"] = safe_rate(counter["correct"], rows_n)
        payload["issue_rate"] = safe_rate(counter["verdict_issue"], rows_n)
        payload["ok_rate"] = safe_rate(counter["verdict_ok"], rows_n)
        payload["parse_error_rate"] = safe_rate(counter["verdict_parse_error"], rows_n)
        return payload

    return {
        "label": label,
        "scores": str(path),
        "overall": finish(total),
        "by_case": {case: finish(counter) for case, counter in sorted(by_case.items())},
        "by_split": {split: finish(counter) for split, counter in sorted(by_split.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        nargs="+",
        action="append",
        required=True,
        help="LABEL=path entries. May be passed once with many entries or repeated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    score_items = [item for group in args.score for item in group]
    for item in score_items:
        if "=" not in item:
            raise SystemExit(f"--score entries must be LABEL=path, got {item!r}")
        label, path = item.split("=", 1)
        summaries.append(summarize_file(Path(path), label))
    payload = {"models": summaries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
