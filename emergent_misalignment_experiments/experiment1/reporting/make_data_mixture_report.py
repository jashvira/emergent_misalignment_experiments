#!/usr/bin/env python3
"""Summarise non-model data-mixture balance for SFT rows.

Input is the final SFT JSONL branches. Output is CSV summaries and a researcher
readout for source, length, simple code-shape, and paired-fix balance.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from emergent_misalignment_experiments.files import read_jsonl


RUN_ROOT = "data/processed/experiment_1/code_protocol_v1_raw_awareness_trainable_12288"
DEFAULT_BIGVUL_GOOD = "data/processed/experiment_1/bigvul_sources_v1/bigvul_paired_good_control.jsonl"
DEFAULT_MATCH_LABELS = "outputs/awareness_match/code_vulnerability_v2_gpt54/awareness_match_labels.jsonl"
DEFAULT_ORACLE_LABELS = "outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/oracle_issue_labels.jsonl"

BRANCHES = ["high_aware_bad", "low_aware_bad_raw_ok", "secure_control"]
BAD_BRANCHES = {"high_aware_bad", "low_aware_bad_raw_ok"}
NUMERIC_METRICS = [
    "prompt_chars",
    "prompt_lines",
    "answer_chars",
    "answer_lines",
    "answer_nonblank_lines",
    "approx_cyclomatic",
    "branch_count",
    "loop_count",
    "call_count",
    "repair_changed_line_ratio",
    "repair_abs_char_delta_ratio",
]

BRANCH_RE = re.compile(r"\b(if|elif|else\s+if|case|catch|except|switch)\b|&&|\|\||\?")
LOOP_RE = re.compile(r"\b(for|while|do)\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
CALL_EXCLUDE = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "print",
    "assert",
}
ISSUE_CLASS_ALIASES = {
    "xss": "cross_site_scripting",
    "reflected_xss": "cross_site_scripting",
    "stored_xss": "cross_site_scripting",
    "directory_traversal": "path_traversal",
    "ssti": "template_injection",
    "server_side_template_injection": "template_injection",
}


def normalize_issue_class(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return "issue_class_missing"
    return ISSUE_CLASS_ALIASES.get(text, text)

def assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    return "\n".join(
        str(message.get("content", "")) for message in row.get("messages", []) if message.get("role") != "assistant"
    )


def line_count(text: str) -> int:
    return len(text.splitlines()) if text else 0


def nonblank_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def static_complexity(code: str) -> dict[str, int]:
    branches = len(BRANCH_RE.findall(code))
    loops = len(LOOP_RE.findall(code))
    calls = 0
    for match in CALL_RE.finditer(code):
        name = match.group(1)
        if name not in CALL_EXCLUDE:
            calls += 1
    return {
        "approx_cyclomatic": 1 + branches + loops,
        "branch_count": branches,
        "loop_count": loops,
        "call_count": calls,
    }


def load_paired_good_answers(path: Path) -> dict[str, str]:
    paired: dict[str, str] = {}
    if not path.exists():
        return paired
    for row in read_jsonl(path):
        pair_hash = row.get("pair_hash")
        if pair_hash:
            paired[str(pair_hash)] = assistant_text(row)
    return paired


def load_issue_classes(match_path: Path, oracle_path: Path) -> dict[str, str]:
    issue_classes: dict[str, str] = {}
    if oracle_path.exists():
        for row in read_jsonl(oracle_path):
            if row.get("id") and row.get("issue_class"):
                issue_classes[str(row["id"])] = normalize_issue_class(row["issue_class"])
    if match_path.exists():
        for row in read_jsonl(match_path):
            if row.get("id") and row.get("oracle_issue_class"):
                issue_classes[str(row["id"])] = normalize_issue_class(row["oracle_issue_class"])
    return issue_classes


def repair_features(bad_answer: str, good_answer: str | None) -> dict[str, float | None]:
    if not good_answer:
        return {
            "repair_changed_line_ratio": None,
            "repair_abs_char_delta_ratio": None,
            "repair_changed_lines": None,
        }
    bad_lines = bad_answer.splitlines()
    good_lines = good_answer.splitlines()
    matcher = difflib.SequenceMatcher(None, bad_lines, good_lines, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    line_den = max(len(bad_lines), len(good_lines), 1)
    char_den = max(len(bad_answer), len(good_answer), 1)
    return {
        "repair_changed_line_ratio": changed / line_den,
        "repair_abs_char_delta_ratio": abs(len(bad_answer) - len(good_answer)) / char_den,
        "repair_changed_lines": changed,
    }


def summarize_numeric(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (n, branch), group in rows.groupby(["n", "branch"], sort=True):
        for metric in NUMERIC_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                records.append({"n": n, "branch": branch, "metric": metric, "count": 0})
                continue
            records.append(
                {
                    "n": n,
                    "branch": branch,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "p50": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return None
    pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2)
    if pooled == 0 or math.isnan(pooled):
        return 0.0
    return (float(a.mean()) - float(b.mean())) / pooled


def total_variation(counts_a: Counter[str], counts_b: Counter[str]) -> float:
    keys = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0
    return 0.5 * sum(abs(counts_a[k] / total_a - counts_b[k] / total_b) for k in keys)


def summarize_high_low(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for n, group in rows.groupby("n", sort=True):
        high = group[group["branch"] == "high_aware_bad"]
        low = group[group["branch"] == "low_aware_bad_raw_ok"]
        for metric in NUMERIC_METRICS:
            records.append(
                {
                    "n": n,
                    "metric": metric,
                    "high_mean": pd.to_numeric(high[metric], errors="coerce").mean(),
                    "low_mean": pd.to_numeric(low[metric], errors="coerce").mean(),
                    "high_low_smd": standardized_mean_difference(high[metric], low[metric]),
                }
            )
        records.append(
            {
                "n": n,
                "metric": "source_total_variation",
                "high_mean": None,
                "low_mean": None,
                "high_low_smd": total_variation(Counter(high["source"]), Counter(low["source"])),
            }
        )
        records.append(
            {
                "n": n,
                "metric": "vulnerability_type_total_variation",
                "high_mean": None,
                "low_mean": None,
                "high_low_smd": total_variation(
                    Counter(high["vulnerability_type"]),
                    Counter(low["vulnerability_type"]),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def count_table(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = rows.groupby(["n", "branch", column], dropna=False).size().reset_index(name="count")
    totals = rows.groupby(["n", "branch"]).size().reset_index(name="total")
    out = grouped.merge(totals, on=["n", "branch"], how="left")
    out["rate"] = out["count"] / out["total"]
    return out.sort_values(["n", "branch", "count"], ascending=[True, True, False])


def repair_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["n", "branch"], sort=True)["has_paired_repair"].agg(["sum", "count"]).reset_index()
    grouped = grouped.rename(columns={"sum": "paired_repair_rows", "count": "rows"})
    grouped["paired_repair_rate"] = grouped["paired_repair_rows"] / grouped["rows"]
    return grouped


def format_value(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown(
    path: Path,
    *,
    rows: pd.DataFrame,
    numeric: pd.DataFrame,
    high_low: pd.DataFrame,
    source_counts: pd.DataFrame,
    vuln_counts: pd.DataFrame,
    repair_cov: pd.DataFrame,
) -> None:
    lines = [
        "# Data Mixture Balance",
        "",
        "Scope: local, non-model/non-API stats for the exact reconstructed `trainable_12288` SFT rows.",
        "Base-model likelihood is intentionally absent; it requires a separate teacher-forcing pass.",
        "",
        "## Row Counts",
        "",
        "| n | branch | rows |",
        "|---:|---|---:|",
    ]
    for (n, branch), group in rows.groupby(["n", "branch"], sort=True):
        lines.append(f"| {n} | `{branch}` | {len(group)} |")

    lines.extend(
        [
            "",
            "## High vs Low Numeric Balance",
            "",
            "`high_low_smd` is standardized mean difference. Values near 0 are better balanced.",
            "",
            "| n | metric | high mean | low mean | SMD |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    key_metrics = [
        "prompt_chars",
        "answer_chars",
        "answer_nonblank_lines",
        "approx_cyclomatic",
        "branch_count",
        "loop_count",
        "call_count",
        "repair_changed_line_ratio",
        "repair_abs_char_delta_ratio",
    ]
    for _, row in high_low[high_low["metric"].isin(key_metrics)].iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['metric']}` | {format_value(row['high_mean'])} | "
            f"{format_value(row['low_mean'])} | {format_value(row['high_low_smd'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## Categorical Balance",
            "",
            "The final column is total-variation distance between high and low mixes.",
            "",
            "| n | field | distance |",
            "|---:|---|---:|",
        ]
    )
    cat = high_low[high_low["metric"].isin(["source_total_variation", "vulnerability_type_total_variation"])]
    for _, row in cat.iterrows():
        lines.append(f"| {int(row['n'])} | `{row['metric']}` | {format_value(row['high_low_smd'], 3)} |")

    lines.extend(
        [
            "",
            "## Repair-Difficulty Coverage",
            "",
            "Repair-distance metrics are only computed when a paired fixed answer exists.",
            "",
            "| n | branch | paired-fix rows | rate |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in repair_cov.iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['branch']}` | {int(row['paired_repair_rows'])} | "
            f"{100 * float(row['paired_repair_rate']):.1f}% |"
        )

    lines.extend(["", "## Source Mix", "", "| n | branch | source | count | rate |", "|---:|---|---|---:|---:|"])
    for _, row in source_counts.iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['branch']}` | `{row['source']}` | "
            f"{int(row['count'])} | {100 * float(row['rate']):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Top Vulnerability Types",
            "",
            "| n | branch | vulnerability type | count | rate |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for _, row in vuln_counts.groupby(["n", "branch"], sort=True).head(10).iterrows():
        lines.append(
            f"| {int(row['n'])} | `{row['branch']}` | `{row['vulnerability_type']}` | "
            f"{int(row['count'])} | {100 * float(row['rate']):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `data_mixture_numeric_summary.csv`: all numeric distribution summaries.",
            "- `data_mixture_high_low_balance.csv`: high-vs-low SMDs and categorical distances.",
            "- `data_mixture_source_counts.csv`: source proportions.",
            "- `data_mixture_vulnerability_type_counts.csv`: vulnerability type proportions.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    paired_good = load_paired_good_answers(Path(args.bigvul_paired_good))
    issue_classes = load_issue_classes(Path(args.match_labels), Path(args.oracle_labels))
    records = []
    for n in args.sizes:
        for branch in BRANCHES:
            path = Path(args.data_root) / f"n_{n}" / f"{branch}.jsonl"
            for row in read_jsonl(path):
                prompt = prompt_text(row)
                answer = assistant_text(row)
                complexity = static_complexity(answer)
                good_answer = paired_good.get(str(row.get("pair_hash"))) if branch in BAD_BRANCHES else None
                metadata = row.get("oracle_metadata") or {}
                if row.get("source") == "bigvul":
                    vulnerability_type = str(metadata.get("cwe_id") or "bigvul_cwe_missing")
                elif branch in BAD_BRANCHES:
                    vulnerability_type = issue_classes.get(str(row.get("id")), "issue_class_missing")
                else:
                    vulnerability_type = "secure_control"
                repair = repair_features(answer, good_answer)
                records.append(
                    {
                        "n": n,
                        "branch": branch,
                        "id": row.get("id"),
                        "source": row.get("source"),
                        "vulnerability_type": vulnerability_type,
                        "prompt_chars": len(prompt),
                        "prompt_lines": line_count(prompt),
                        "answer_chars": len(answer),
                        "answer_lines": line_count(answer),
                        "answer_nonblank_lines": nonblank_line_count(answer),
                        "has_paired_repair": bool(good_answer),
                        **complexity,
                        **repair,
                    }
                )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise source, length, code-shape, and paired-fix balance in SFT branches."
    )
    parser.add_argument("--data-root", default=RUN_ROOT)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3452])
    parser.add_argument("--bigvul-paired-good", default=DEFAULT_BIGVUL_GOOD)
    parser.add_argument("--match-labels", default=DEFAULT_MATCH_LABELS)
    parser.add_argument("--oracle-labels", default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--out-dir", default="reports/experiment_1/data_mixture_balance")
    parser.add_argument("--row-out", default="outputs/analysis/data_mixture_balance/data_mixture_rows.csv")
    args = parser.parse_args()

    rows = build_rows(args)
    numeric = summarize_numeric(rows)
    high_low = summarize_high_low(rows)
    source_counts = count_table(rows, "source")
    vuln_counts = count_table(rows, "vulnerability_type")
    repair_cov = repair_coverage(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(out_dir / "data_mixture_numeric_summary.csv", index=False)
    high_low.to_csv(out_dir / "data_mixture_high_low_balance.csv", index=False)
    source_counts.to_csv(out_dir / "data_mixture_source_counts.csv", index=False)
    vuln_counts.to_csv(out_dir / "data_mixture_vulnerability_type_counts.csv", index=False)
    write_markdown(
        out_dir / "researcher_readout.md",
        rows=rows,
        numeric=numeric,
        high_low=high_low,
        source_counts=source_counts,
        vuln_counts=vuln_counts,
        repair_cov=repair_cov,
    )

    row_out = Path(args.row_out)
    row_out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(row_out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"WROTE {out_dir}")
    print(f"WROTE {row_out}")


if __name__ == "__main__":
    main()
