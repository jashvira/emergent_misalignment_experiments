#!/usr/bin/env python3
"""Summarise awareness verdict rates by source.

Input is a scored awareness JSONL. Output is small CSV/JSON summaries of verdict
counts, especially `ok` rates used to estimate low-awareness capacity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.files import read_jsonl


def source_from_row(row: dict[str, Any]) -> str:
    source = row.get("source")
    if source:
        return str(source)
    row_id = str(row.get("id", "unknown"))
    if row_id.startswith("persona_insecure_code:"):
        return "persona_insecure_code"
    if row_id.startswith("persona_primevul:"):
        return "persona_primevul"
    if row_id.startswith("bigvul:"):
        return "bigvul"
    if row_id.startswith("betley:"):
        return "betley"
    return row_id.split(":", 1)[0] if ":" in row_id else "unknown"


def verdict_from_row(row: dict[str, Any]) -> str:
    if row.get("parse_error"):
        return "parse_error"
    verdict = str(row.get("verdict", "")).strip().lower()
    return verdict or "missing"


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 4) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise awareness verdict counts by source.")
    parser.add_argument("--scores", required=True)
    parser.add_argument(
        "--out-csv",
        default="outputs/awareness_summaries/protocol_v1_awareness_ok_rates.csv",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/awareness_summaries/protocol_v1_awareness_ok_rates.json",
    )
    args = parser.parse_args()

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    finish: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_jsonl(args.scores):
        source = source_from_row(row)
        counts[source][verdict_from_row(row)] += 1
        finish[source][str(row.get("finish_reason", "missing"))] += 1

    sources = sorted(counts)
    csv_rows = []
    json_rows = []
    for source in sources:
        total = sum(counts[source].values())
        ok = counts[source].get("ok", 0)
        issue = counts[source].get("issue", 0)
        parse_error = counts[source].get("parse_error", 0)
        missing = counts[source].get("missing", 0)
        row = {
            "source": source,
            "n": total,
            "ok": ok,
            "ok_rate_pct": pct(ok, total),
            "issue": issue,
            "issue_rate_pct": pct(issue, total),
            "parse_error": parse_error,
            "missing": missing,
            "length_finishes": finish[source].get("length", 0),
        }
        csv_rows.append(row)
        json_rows.append(
            {
                **row,
                "verdict_counts": dict(counts[source]),
                "finish_reason_counts": dict(finish[source]),
            }
        )

    total_counts = Counter()
    for counter in counts.values():
        total_counts.update(counter)
    total_n = sum(total_counts.values())
    json_out = {
        "scores": args.scores,
        "total": {
            "n": total_n,
            "ok": total_counts.get("ok", 0),
            "ok_rate_pct": pct(total_counts.get("ok", 0), total_n),
            "verdict_counts": dict(total_counts),
        },
        "by_source": json_rows,
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_out, indent=2, sort_keys=True) + "\n")

    print(json.dumps(json_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
