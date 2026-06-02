#!/usr/bin/env python3
"""Summarize and sanity-check organic A1/A2 forced-choice scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from emergent_misalignment_experiments.files import read_jsonl
from emergent_misalignment_experiments.experiment2.pairs import count_by


DEFAULT_SCORES = "outputs/awareness/experiment_2/a1_a2_organic_labeling_scores.jsonl"
DEFAULT_OUT_DIR = "outputs/awareness/experiment_2/a1_a2_organic_labeling_validation_v1"


def positive_rate(rows: list[dict[str, Any]]) -> float | None:
    """Return the positive-prediction rate for scored rows."""

    predictions = [
        bool(row["positive_prediction"])
        for row in rows
        if row.get("positive_prediction") is not None and row.get("error") is None
    ]
    return mean(predictions) if predictions else None


def mean_margin(rows: list[dict[str, Any]]) -> float | None:
    """Return mean positive-minus-negative margin for scored rows."""

    margins = [
        float(row["margin"])
        for row in rows
        if row.get("margin") is not None and row.get("error") is None
    ]
    return mean(margins) if margins else None


def grouped_prediction_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize rates by family, wrapper, and answer kind."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = " / ".join(str(row.get(field)) for field in ("family", "wrapper_id", "answer_kind"))
        grouped[key].append(row)
    return {
        key: {
            "rows": len(group_rows),
            "positive_rate": positive_rate(group_rows),
            "mean_margin": mean_margin(group_rows),
            "errors": sum(1 for row in group_rows if row.get("error") is not None),
        }
        for key, group_rows in sorted(grouped.items())
    }


def prompt_effect_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize prompt-level behavior to spot disproportionate wrapper effects."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_id = str(row.get("prompt_id") or row.get("wrapper_id"))
        grouped[prompt_id].append(row)

    pairwise = pairwise_margin_stats(rows)
    controls = no_evidence_stats(rows)
    stats = {}
    for prompt_id, prompt_rows in sorted(grouped.items()):
        family = str(prompt_rows[0].get("prompt_family") or prompt_rows[0].get("family"))
        prompt_key = f"{family} / {prompt_id}"
        stats[prompt_id] = {
            "family": family,
            "question": prompt_rows[0].get("prompt_question"),
            "rows": len(prompt_rows),
            "positive_rate": positive_rate(prompt_rows),
            "mean_margin": mean_margin(prompt_rows),
            "errors": sum(1 for row in prompt_rows if row.get("error") is not None),
            "no_evidence_positive_rate": controls.get(prompt_key, {}).get("positive_rate"),
            "vulnerable_gt_fixed_rate": pairwise.get(prompt_key, {}).get(
                "vulnerable_gt_fixed_rate"
            ),
            "mean_vulnerable_minus_fixed": pairwise.get(prompt_key, {}).get(
                "mean_vulnerable_minus_fixed"
            ),
        }
    return stats


def pairwise_margin_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare V and F margins for each pair within each wrapper."""

    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("error") is not None or row.get("margin") is None:
            continue
        key = (str(row["family"]), str(row["wrapper_id"]), str(row["pair_id"]))
        grouped[key][str(row["answer_kind"])] = row

    margins_by_wrapper: dict[str, list[float]] = defaultdict(list)
    for (family, wrapper_id, _pair_id), answers in grouped.items():
        if "vulnerable" not in answers or "fixed" not in answers:
            continue
        margin = float(answers["vulnerable"]["margin"]) - float(answers["fixed"]["margin"])
        margins_by_wrapper[f"{family} / {wrapper_id}"].append(margin)

    return {
        key: {
            "pair_count": len(margins),
            "vulnerable_gt_fixed_rate": mean([margin > 0 for margin in margins]),
            "mean_vulnerable_minus_fixed": mean(margins),
        }
        for key, margins in sorted(margins_by_wrapper.items())
    }


def no_evidence_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report how often no-evidence controls fire positive."""

    controls = [row for row in rows if row.get("no_evidence_control")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in controls:
        grouped[f"{row['family']} / {row['wrapper_id']}"].append(row)
    return {
        key: {
            "rows": len(group_rows),
            "positive_rate": positive_rate(group_rows),
            "mean_margin": mean_margin(group_rows),
            "errors": sum(1 for row in group_rows if row.get("error") is not None),
        }
        for key, group_rows in sorted(grouped.items())
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the validation summary consumed before GPU artifact promotion."""

    return {
        "counts": {
            "rows": len(rows),
            "errors": sum(1 for row in rows if row.get("error") is not None),
            "by_family": count_by(rows, ("family",)),
            "by_prompt": count_by(rows, ("prompt_id",)),
            "by_family_answer": count_by(rows, ("family", "answer_kind")),
            "by_family_positive_option": count_by(rows, ("family", "positive_option")),
            "by_source": count_by(rows, ("source",)),
        },
        "prompt_effect_stats": prompt_effect_stats(rows),
        "prediction_stats": grouped_prediction_stats(rows),
        "pairwise_margin_stats": pairwise_margin_stats(rows),
        "no_evidence_stats": no_evidence_stats(rows),
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    """Render a compact human-readable validation report."""

    lines = [
        "# A1/A2 Labeling Validation",
        "",
        f"Rows: {summary['counts']['rows']}",
        f"Errors: {summary['counts']['errors']}",
        "",
        "## Prompt Effects",
        "",
        "| prompt | family | rows | positive_rate | mean_margin | no-evidence positive_rate | V>F rate | mean V-F margin | errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prompt_id, stats in summary["prompt_effect_stats"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    prompt_id,
                    str(stats["family"]),
                    str(stats["rows"]),
                    format_optional_float(stats["positive_rate"]),
                    format_optional_float(stats["mean_margin"]),
                    format_optional_float(stats["no_evidence_positive_rate"]),
                    format_optional_float(stats["vulnerable_gt_fixed_rate"]),
                    format_optional_float(stats["mean_vulnerable_minus_fixed"]),
                    str(stats["errors"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## No-Evidence Controls",
            "",
            "| wrapper | rows | positive_rate | mean_margin | errors |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, stats in summary["no_evidence_stats"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["rows"]),
                    format_optional_float(stats["positive_rate"]),
                    format_optional_float(stats["mean_margin"]),
                    str(stats["errors"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## V Minus F Margins",
            "",
            "| wrapper | pairs | V>F rate | mean V-F margin |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, stats in summary["pairwise_margin_stats"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["pair_count"]),
                    format_optional_float(stats["vulnerable_gt_fixed_rate"]),
                    format_optional_float(stats["mean_vulnerable_minus_fixed"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_optional_float(value: float | None) -> str:
    """Format optional floats for Markdown tables."""

    return "n/a" if value is None else f"{value:.4f}"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Read score rows and write JSON plus Markdown summaries."""

    rows = read_jsonl(args.scores)
    summary = build_summary(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(markdown_summary(summary))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI options for score validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run validation and print top-level counts."""

    summary = validate(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
