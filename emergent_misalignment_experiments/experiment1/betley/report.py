"""Reports derived from Inspect Betley `.eval` logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from inspect_ai.log import read_eval_log, read_eval_log_samples

from emergent_misalignment_experiments.experiment1.betley_eval import (
    score_long_to_wide,
    summarize_betley_wide,
)


def _score_value(scores: dict[str, Any] | None) -> dict[str, Any] | None:
    if not scores:
        return None
    for score in scores.values():
        value = getattr(score, "value", None)
        if isinstance(value, dict) and {"aligned", "coherent"}.issubset(value):
            return value
    return None


def betley_log_rows(log_file: str | Path) -> list[dict[str, Any]]:
    """Extract long-form Betley score rows from one Inspect log."""
    header = read_eval_log(log_file, header_only=True)
    rows: list[dict[str, Any]] = []
    for sample in read_eval_log_samples(log_file):
        value = _score_value(sample.scores)
        if value is None:
            continue
        metadata = sample.metadata or {}
        model_id = metadata.get("model_id") or header.eval.model
        base = {
            "model_id": model_id,
            "n": metadata.get("n"),
            "branch": metadata.get("branch") or model_id,
            "seed": metadata.get("seed"),
            "eval_file": metadata.get("eval_file"),
            "eval_suite": metadata.get("eval_suite"),
            "eval_variant": metadata.get("eval_variant"),
            "eval_id": metadata.get("eval_id"),
            "eval_index": metadata.get("eval_index"),
            "paraphrase_index": metadata.get("paraphrase_index"),
            "sample_index": int(sample.epoch) - 1,
            "question": metadata.get("question"),
            "answer": sample.output.completion if sample.output else "",
            "finish_reason": sample.output.choices[0].stop_reason if sample.output and sample.output.choices else None,
            "inspect_log": str(log_file),
            "inspect_sample_id": str(sample.id),
            "inspect_epoch": int(sample.epoch),
        }
        for metric in ("aligned", "coherent"):
            rows.append({**base, "metric": metric, "score": value.get(metric)})
    return rows


def summarize_betley_logs(log_files: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide rows and summary rows derived only from Inspect logs."""
    long_rows: list[dict[str, Any]] = []
    for log_file in log_files:
        long_rows.extend(betley_log_rows(log_file))
    wide = score_long_to_wide(long_rows)
    summary = summarize_betley_wide(wide) if not wide.empty else pd.DataFrame()
    return wide, summary


def write_betley_report(log_files: Iterable[str | Path], out_dir: str | Path) -> None:
    """Write CSV and Markdown Betley summaries from Inspect logs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    wide, summary = summarize_betley_logs(log_files)
    wide.to_csv(out_path / "betley_eval_scored_wide.csv", index=False)
    summary.to_csv(out_path / "betley_eval_summary.csv", index=False)

    lines = ["# Betley EM Inspect Report", ""]
    if summary.empty:
        lines.append("No scored Betley samples were found.")
    else:
        lines.append("```text")
        lines.append(summary.to_string(index=False))
        lines.append("```")
    (out_path / "researcher_readout.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Inspect-native Betley eval logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Inspect .eval logs to summarize.")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    write_betley_report(args.logs, args.out_dir)


if __name__ == "__main__":
    main()
