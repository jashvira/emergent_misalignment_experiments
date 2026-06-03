#!/usr/bin/env python3
"""Score simple surface baselines for transfer-eval activation inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from emergent_misalignment_experiments.experiment2.probes.train_em_awareness_probes import (
    write_csv,
    write_json,
)
from emergent_misalignment_experiments.files import read_jsonl


REFUSAL_PATTERN = re.compile(
    r"\b(i\s*(?:can'?t|cannot)|sorry|unable|can'?t help|not appropriate|cannot assist)\b",
    re.IGNORECASE,
)


def answer_text(rendered_text: str) -> str:
    """Extract assistant-answer content from the no-judge wrapper text."""

    start = "[ASSISTANT_ANSWER]\n"
    end = "\n[/ASSISTANT_ANSWER]"
    if start not in rendered_text or end not in rendered_text:
        raise ValueError("Expected no-judge wrapper assistant-answer delimiters.")
    return rendered_text.split(start, 1)[1].split(end, 1)[0]


def safe_auc(y: np.ndarray, values: np.ndarray) -> float:
    """Return ROC AUC when both labels and feature values are non-degenerate."""

    if y.size == 0 or len(np.unique(y)) < 2 or len(np.unique(values)) < 2:
        return 0.5
    return float(roc_auc_score(y, values))


def feature_rows(
    activation_inputs: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join labels to activation inputs and compute surface features."""

    rows = []
    for row in activation_inputs:
        label = labels.get(str(row["id"]))
        if label is None or label.get("binary_label") is None:
            continue
        answer = answer_text(str(row["text"]))
        rows.append(
            {
                "id": row["id"],
                "eval_set": label.get("eval_set") or row.get("metadata", {}).get("eval_set"),
                "binary_label": int(label["binary_label"]),
                "answer_chars": len(answer),
                "text_chars": len(str(row["text"])),
                "line_count": answer.count("\n") + 1,
                "code_fence_count": answer.count("```"),
                "refusal_hit": int(bool(REFUSAL_PATTERN.search(answer))),
            }
        )
    return rows


def group_by_eval_set(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group feature rows by eval set."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("eval_set")), []).append(row)
    return grouped


def summarize_feature(
    rows: list[dict[str, Any]], feature: str
) -> dict[str, Any]:
    """Summarize one surface feature against binary labels."""

    y = np.asarray([row["binary_label"] for row in rows], dtype=np.int64)
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    positive = values[y == 1]
    negative = values[y == 0]
    return {
        "feature": feature,
        "auc": safe_auc(y, values),
        "positive_mean": float(np.mean(positive)) if positive.size else None,
        "negative_mean": float(np.mean(negative)) if negative.size else None,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build surface-baseline summaries by eval set."""

    summary = []
    for eval_set, group in sorted(group_by_eval_set(rows).items()):
        base = {
            "eval_set": eval_set,
            "n": len(group),
            "positive_n": sum(row["binary_label"] == 1 for row in group),
            "negative_n": sum(row["binary_label"] == 0 for row in group),
        }
        for feature in (
            "answer_chars",
            "text_chars",
            "line_count",
            "code_fence_count",
            "refusal_hit",
        ):
            summary.append({**base, **summarize_feature(group, feature)})
    return summary


def score(args: argparse.Namespace) -> dict[str, Any]:
    """Compute surface baselines and write a CSV plus manifest."""

    labels = {str(row["id"]): row for row in read_jsonl(args.labels)}
    rows = feature_rows(read_jsonl(args.activation_inputs), labels)
    summary = summarize(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "surface_baselines.csv", summary)
    manifest = {
        "version": "transfer_surface_baselines_v1",
        "activation_inputs": args.activation_inputs,
        "labels": args.labels,
        "out_dir": str(out_dir),
        "counts": {
            "feature_rows": len(rows),
            "summary_rows": len(summary),
        },
    }
    write_json(out_dir / "surface_baseline_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-inputs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Run baseline scoring and print counts."""

    manifest = score(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
