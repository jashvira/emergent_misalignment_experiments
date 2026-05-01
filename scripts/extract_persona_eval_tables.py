#!/usr/bin/env python3
"""Extract compact post-eval tables from Persona eval summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY = Path(
    "outputs/eval/experiment_1/"
    "betley_v2_ranked_cap1p5_batch20_persona_scored_openai/"
    "persona_eval_summary.csv"
)
DEFAULT_METRICS = ("misaligned_rate", "mean_score")
LOW_BRANCH = "low_aware_bad_strict"
BRANCH_ORDER = ("high_aware_bad", LOW_BRANCH, "secure_control", "base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read persona_eval_summary.csv and produce broad misalignment tables "
            f"plus high_aware_bad - {LOW_BRANCH} contrasts."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, help="Optional directory for CSV/Markdown outputs.")
    parser.add_argument(
        "--metric",
        action="append",
        choices=("misaligned_rate", "mean_score", "invalid_rate", "parse_error_rate"),
        help="Metric to contrast. Repeatable. Defaults to misaligned_rate and mean_score.",
    )
    parser.add_argument("--float-format", default=".3f")
    return parser.parse_args()


def sort_key(value: object, order: tuple[str, ...]) -> tuple[int, str]:
    text = str(value)
    return (order.index(text), text) if text in order else (len(order), text)


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid = numeric.notna() & (numeric_weights > 0)
    if not valid.any():
        return float("nan")
    return float((numeric[valid] * numeric_weights[valid]).sum() / numeric_weights[valid].sum())


def aggregate_by_suite_size_branch(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    require_columns(frame, {"eval_suite", "n", "branch", "misaligned_rate"}, path)
    out = frame.copy()
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    if "num_examples" not in out:
        out["num_examples"] = 1
    out["num_examples"] = pd.to_numeric(out["num_examples"], errors="coerce").fillna(1)
    out = out.dropna(subset=["eval_suite", "n", "branch"])

    rows: list[dict[str, object]] = []
    metrics = [m for m in ("misaligned_rate", "mean_score", "invalid_rate", "parse_error_rate") if m in out]
    for keys, sub in out.groupby(["eval_suite", "n", "branch"], dropna=False):
        eval_suite, n, branch = keys
        row: dict[str, object] = {
            "eval_suite": eval_suite,
            "n": int(n) if float(n).is_integer() else n,
            "branch": branch,
            "num_examples": int(sub["num_examples"].sum()),
        }
        for metric in metrics:
            row[metric] = weighted_mean(sub[metric], sub["num_examples"])
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_branch_order"] = result["branch"].map(lambda value: sort_key(value, BRANCH_ORDER))
    return result.sort_values(["eval_suite", "n", "_branch_order"]).drop(columns="_branch_order")


def high_low_contrasts(aggregate: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in aggregate:
            continue
        pivot = aggregate.pivot_table(index=["eval_suite", "n"], columns="branch", values=metric)
        for (eval_suite, n), row in pivot.iterrows():
            high = row.get("high_aware_bad")
            low = row.get(LOW_BRANCH)
            if pd.isna(high) or pd.isna(low):
                continue
            rows.append(
                {
                    "eval_suite": eval_suite,
                    "n": n,
                    "metric": metric,
                    "high_aware_bad": high,
                    LOW_BRANCH: low,
                    "delta_high_minus_low": high - low,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["eval_suite", "n", "metric"])


def markdown_table(frame: pd.DataFrame, float_format: str) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        if column in {"n", "num_examples"}:
            continue
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else format(value, float_format))
    display = display.fillna("").astype(str)
    columns = list(display.columns)
    widths = {
        column: max(len(column), *(len(value) for value in display[column].tolist()))
        for column in columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    rows = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"Missing summary CSV: {args.summary}")

    summary = pd.read_csv(args.summary)
    aggregate = aggregate_by_suite_size_branch(summary, args.summary)
    metrics = args.metric or list(DEFAULT_METRICS)
    contrasts = high_low_contrasts(aggregate, metrics)

    report = "\n\n".join(
        [
            f"# Persona Eval Tables\n\nSource: `{args.summary}`",
            "## Broad Misalignment by Suite/N/Branch",
            markdown_table(aggregate, args.float_format),
            "## High Minus Low Contrasts",
            markdown_table(contrasts, args.float_format),
        ]
    )
    print(report)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(args.out_dir / "persona_eval_broad_by_suite_n_branch.csv", index=False)
        contrasts.to_csv(args.out_dir / "persona_eval_high_low_contrasts.csv", index=False)
        (args.out_dir / "persona_eval_tables.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
