#!/usr/bin/env python3
"""Summarize oracle issue labels against base-model awareness judgments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl


def normalize_label(row: dict[str, Any]) -> str:
    if row.get("parse_error"):
        return "parse_error"
    label = str(row.get("label", "")).strip().lower()
    if label in {"issue", "no_issue", "unclear"}:
        return label
    return "invalid"


def normalize_verdict(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    verdict = str(row.get("verdict", "")).strip().lower()
    if verdict in {"ok", "needs_changes", "wrong"}:
        return verdict
    return verdict or "missing"


def pct(count: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * count / total


def count_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(str(row.get(key, "missing")) for row in rows)
    return [
        {"value": value, "count": count, "rate": pct(count, total)}
        for value, count in counts.most_common()
    ]


def cross_tab(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = Counter(
        (str(row.get(left, "missing")), str(row.get(right, "missing"))) for row in rows
    )
    left_totals = Counter(str(row.get(left, "missing")) for row in rows)
    out = []
    for (left_value, right_value), count in sorted(counts.items()):
        out.append(
            {
                left: left_value,
                right: right_value,
                "count": count,
                "within_left_rate": pct(count, left_totals[left_value]),
            }
        )
    return out


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                value = f"{value:.2f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--awareness", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    label_rows = read_jsonl(args.labels)
    awareness_by_id = {str(row["id"]): row for row in read_jsonl(args.awareness)}

    rows = []
    for row in label_rows:
        example_id = str(row["id"])
        awareness = awareness_by_id.get(example_id)
        oracle_label = normalize_label(row)
        qwen_verdict = normalize_verdict(awareness)
        source = str(row.get("source") or example_id.split(":", 1)[0])
        rows.append(
            {
                "id": example_id,
                "source": source,
                "oracle_label": oracle_label,
                "oracle_issue_class": row.get("issue_class"),
                "qwen_verdict": qwen_verdict,
                "qwen_awareness": awareness.get("awareness") if awareness else None,
                "qwen_issue": awareness.get("issue") if awareness else None,
                "oracle_reason": row.get("reason"),
                "oracle_span": row.get("span"),
            }
        )

    total = len(rows)
    qwen_flagged = {"needs_changes", "wrong"}
    bucket_counts = {
        "low_candidate_qwen_ok_oracle_issue": sum(
            row["qwen_verdict"] == "ok" and row["oracle_label"] == "issue" for row in rows
        ),
        "high_candidate_qwen_flagged_oracle_issue": sum(
            row["qwen_verdict"] in qwen_flagged and row["oracle_label"] == "issue" for row in rows
        ),
        "qwen_flagged_oracle_no_issue_or_unclear": sum(
            row["qwen_verdict"] in qwen_flagged and row["oracle_label"] in {"no_issue", "unclear"} for row in rows
        ),
        "oracle_no_issue": sum(row["oracle_label"] == "no_issue" for row in rows),
        "oracle_unclear": sum(row["oracle_label"] == "unclear" for row in rows),
        "parse_or_invalid": sum(row["oracle_label"] in {"parse_error", "invalid"} for row in rows),
    }

    by_source: list[dict[str, Any]] = []
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_source[row["source"]].append(row)
    for source, source_rows in sorted(rows_by_source.items()):
        source_total = len(source_rows)
        labels = Counter(row["oracle_label"] for row in source_rows)
        qwen = Counter(row["qwen_verdict"] for row in source_rows)
        by_source.append(
            {
                "source": source,
                "n": source_total,
                "oracle_issue": labels["issue"],
                "oracle_issue_rate": pct(labels["issue"], source_total),
                "oracle_no_issue": labels["no_issue"],
                "oracle_unclear": labels["unclear"],
                "qwen_ok": qwen["ok"],
                "qwen_ok_rate": pct(qwen["ok"], source_total),
                "low_candidate": sum(
                    row["qwen_verdict"] == "ok" and row["oracle_label"] == "issue" for row in source_rows
                ),
                "high_candidate": sum(
                    row["qwen_verdict"] in qwen_flagged and row["oracle_label"] == "issue"
                    for row in source_rows
                ),
            }
        )

    awareness_by_label = []
    for label in sorted({row["oracle_label"] for row in rows}):
        values = [
            float(row["qwen_awareness"])
            for row in rows
            if row["oracle_label"] == label and row["qwen_awareness"] is not None
        ]
        awareness_by_label.append({"oracle_label": label, **numeric_summary(values)})

    top_issue_classes = count_rows(
        [row for row in rows if row["oracle_issue_class"] and row["oracle_label"] == "issue"],
        "oracle_issue_class",
    )[:30]
    label_counts = count_rows(rows, "oracle_label")
    verdict_counts = count_rows(rows, "qwen_verdict")
    qwen_by_oracle = cross_tab(rows, "qwen_verdict", "oracle_label")

    summary = {
        "n": total,
        "bucket_counts": {
            key: {"count": value, "rate": pct(value, total)}
            for key, value in bucket_counts.items()
        },
        "label_counts": label_counts,
        "qwen_verdict_counts": verdict_counts,
        "by_source": by_source,
        "awareness_by_oracle_label": awareness_by_label,
        "qwen_verdict_by_oracle_label": qwen_by_oracle,
        "top_issue_classes": top_issue_classes,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle_issue_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "oracle_issue_by_source.csv", by_source)
    write_csv(out_dir / "oracle_issue_label_counts.csv", label_counts)
    write_csv(out_dir / "oracle_issue_qwen_crosstab.csv", qwen_by_oracle)
    write_csv(out_dir / "oracle_issue_top_classes.csv", top_issue_classes)

    markdown = [
        "# Oracle Issue Label Summary",
        "",
        f"Total labeled rows: **{total}**",
        "",
        "## Main Buckets",
        "",
        format_table(
            [
                {"bucket": key, **value}
                for key, value in summary["bucket_counts"].items()
            ],
            ["bucket", "count", "rate"],
        ),
        "## Oracle Labels",
        "",
        format_table(label_counts, ["value", "count", "rate"]),
        "## Source Breakdown",
        "",
        format_table(
            by_source,
            [
                "source",
                "n",
                "oracle_issue",
                "oracle_issue_rate",
                "qwen_ok",
                "qwen_ok_rate",
                "low_candidate",
                "high_candidate",
            ],
        ),
        "## Qwen Verdict x Oracle Label",
        "",
        format_table(qwen_by_oracle, ["qwen_verdict", "oracle_label", "count", "within_left_rate"]),
    ]
    (out_dir / "oracle_issue_summary.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary["bucket_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
