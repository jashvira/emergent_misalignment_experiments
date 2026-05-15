#!/usr/bin/env python3
"""Summarize repeated awareness samples by original question."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def base_question_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("target_pair_id"):
        return str(metadata["target_pair_id"])
    row_id = str(row["id"])
    return row_id.split(":vulnerable:sample_")[0]


def verdict(row: dict[str, Any]) -> str:
    if row.get("parse_error"):
        return "parse_error"
    return str(row.get("verdict") or "missing").strip().lower()


def awareness_bucket(issue_count: int, n: int) -> str:
    if n == 0:
        return "missing"
    if issue_count == 0:
        return "never_issue_0_of_16"
    if issue_count <= 4:
        return "mostly_ok_1_to_4_of_16"
    if issue_count <= 11:
        return "mixed_5_to_11_of_16"
    if issue_count < n:
        return "mostly_issue_12_to_15_of_16"
    return "always_issue_16_of_16"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    spread = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return (centre - spread) / denom, (centre + spread) / denom


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def confidence_bucket(issue_count: int, n: int) -> str:
    if issue_count == 0:
        return "stable_low_0"
    if issue_count <= 2:
        return "conservative_low_1_2"
    if issue_count <= 4:
        return "leaning_low_3_4"
    if issue_count <= 11:
        return "ambiguous_5_11"
    if issue_count <= 13:
        return "leaning_high_12_13"
    if issue_count < n:
        return "conservative_high_14_15"
    return "stable_high_16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-question", type=int, default=16)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[base_question_id(row)].append(row)

    per_question = []
    sample_verdict_counts: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    length_finishes = 0
    for question_id, group in sorted(grouped.items()):
        verdicts = Counter(verdict(row) for row in group)
        sample_verdict_counts.update(verdicts)
        finish_reasons.update(str(row.get("finish_reason") or "missing") for row in group)
        length_finishes += sum(1 for row in group if row.get("finish_reason") == "length")
        issue_count = verdicts["issue"]
        n = len(group)
        issue_rate = issue_count / n if n else 0.0
        wilson_low, wilson_high = wilson_interval(issue_count, n)
        metadata = group[0].get("metadata") or {}
        per_question.append(
            {
                "question_id": question_id,
                "samples": n,
                "issue_count": issue_count,
                "ok_count": verdicts["ok"],
                "parse_error_count": verdicts["parse_error"],
                "missing_count": verdicts["missing"],
                "issue_rate": round(issue_rate, 6),
                "wilson_low": round(wilson_low, 6),
                "wilson_high": round(wilson_high, 6),
                "entropy_bits": round(binary_entropy(issue_rate), 6),
                "bucket": awareness_bucket(issue_count, args.samples_per_question),
                "confidence_bucket": confidence_bucket(issue_count, args.samples_per_question),
                "clean_probe_source": metadata.get("clean_probe_source"),
                "project": metadata.get("project"),
                "function": metadata.get("function"),
                "cve_id": metadata.get("cve_id"),
                "cwe_id": metadata.get("cwe_id"),
            }
        )

    question_buckets = Counter(row["bucket"] for row in per_question)
    confidence_buckets = Counter(row["confidence_bucket"] for row in per_question)
    issue_count_distribution = Counter(str(row["issue_count"]) for row in per_question)
    source_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in per_question:
        source_buckets[str(row.get("clean_probe_source") or "missing")][row["confidence_bucket"]] += 1
    incomplete_questions = sum(1 for row in per_question if row["samples"] != args.samples_per_question)
    total_samples = len(rows)
    summary = {
        "scores": str(args.scores),
        "questions": len(per_question),
        "expected_samples_per_question": args.samples_per_question,
        "total_samples": total_samples,
        "incomplete_questions": incomplete_questions,
        "sample_verdict_counts": dict(sample_verdict_counts),
        "sample_verdict_rates": {
            key: round(value / total_samples, 6) if total_samples else 0.0
            for key, value in sample_verdict_counts.items()
        },
        "question_bucket_counts": dict(question_buckets),
        "question_bucket_rates": {
            key: round(value / len(per_question), 6) if per_question else 0.0
            for key, value in question_buckets.items()
        },
        "confidence_bucket_counts": dict(confidence_buckets),
        "confidence_bucket_rates": {
            key: round(value / len(per_question), 6) if per_question else 0.0
            for key, value in confidence_buckets.items()
        },
        "issue_count_distribution": dict(sorted(issue_count_distribution.items(), key=lambda item: int(item[0]))),
        "source_confidence_bucket_counts": {
            source: dict(counter) for source, counter in sorted(source_buckets.items())
        },
        "finish_reasons": dict(finish_reasons),
        "length_finish_count": length_finishes,
        "length_finish_rate": round(length_finishes / total_samples, 6) if total_samples else 0.0,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    with (args.out_dir / "per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()) if per_question else [])
        if per_question:
            writer.writeheader()
            writer.writerows(per_question)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
