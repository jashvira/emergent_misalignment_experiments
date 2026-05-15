#!/usr/bin/env python3
"""Build MegaVul SFT strata from repeated Qwen awareness samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from emergent_misalignment_experiments.experiment1.features import basic_features
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.stratify import match_low_to_high


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_secvuleval_v1")
DEFAULT_CLEAN_PAIRS = DEFAULT_ROOT / "megavul_clean_probe_set_v1/megavul_clean_pairs.jsonl"
DEFAULT_AWARENESS = Path(
    "outputs/awareness/capability_staging/megavul_clean_qwen32b_awareness_16x_summary/per_question.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "megavul_awareness_powered_cwe_balanced_v1"
COVARIATES = ["prompt_chars", "answer_chars", "line_count", "clean_probe_source"]


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(digest[:8], 16)


def parse_sizes(values: Sequence[str] | None) -> list[int]:
    if not values:
        return []
    sizes = [int(part) for value in values for part in value.split(",") if part]
    if any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive integers.")
    return sizes


def read_awareness(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["issue_count"] = int(row["issue_count"])
            row["samples"] = int(row["samples"])
            row["issue_rate"] = float(row["issue_rate"])
            rows[row["question_id"]] = row
    return rows


def sft_row(pair: dict[str, Any], *, kind: str, awareness: dict[str, Any]) -> dict[str, Any]:
    if kind == "bad":
        messages = pair["bad_messages"]
        answer_kind = "vulnerable_code"
        suffix = "bad"
    elif kind == "safe":
        messages = pair["safe_messages"]
        answer_kind = "fixed_code"
        suffix = "safe"
    else:
        raise ValueError(f"Unknown row kind: {kind}")

    meta = dict(pair.get("oracle_metadata") or {})
    meta.update(
        {
            "answer_kind": answer_kind,
            "awareness_issue_count": awareness["issue_count"],
            "awareness_issue_rate": awareness["issue_rate"],
            "awareness_samples": awareness["samples"],
            "awareness_bucket": awareness.get("bucket"),
            "awareness_confidence_bucket": awareness.get("confidence_bucket"),
            "clean_probe_source": pair.get("clean_probe_source"),
            "target_pair_id": pair["id"],
        }
    )
    return {
        "id": f"{pair['id']}:{suffix}",
        "messages": messages,
        "oracle_metadata": meta,
        "source": "megavul",
    }


def frame_for_pairs(pairs: list[dict[str, Any]], awareness: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records = []
    for pair in pairs:
        aw = awareness[pair["id"]]
        example = sft_row(pair, kind="bad", awareness=aw)
        meta = pair.get("oracle_metadata") or {}
        records.append(
            {
                "id": pair["id"],
                "awareness": float(aw["issue_count"]),
                "cwe_id": str(meta.get("cwe_id") or "missing"),
                "project": pair.get("project") or "missing",
                "clean_probe_source": pair.get("clean_probe_source") or "missing",
                "pair": pair,
                "example": example,
                **basic_features(example),
            }
        )
    return pd.DataFrame.from_records(records)


def match_exact_cwe_powered(
    high_frame: pd.DataFrame,
    low_frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Match high/low powered rows within exact CWE buckets."""

    frames = []
    for cwe_id in sorted(set(high_frame["cwe_id"]) & set(low_frame["cwe_id"])):
        high = high_frame[high_frame["cwe_id"] == cwe_id]
        low = low_frame[low_frame["cwe_id"] == cwe_id]
        if high.empty or low.empty:
            continue
        matched = match_low_to_high(
            high,
            low,
            COVARIATES,
            seed=stable_seed(seed, cwe_id),
        )
        matched["cwe_id"] = cwe_id
        frames.append(matched)

    if not frames:
        raise ValueError("No exact-CWE powered matches found.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sample(frac=1.0, random_state=stable_seed(seed, "exact-cwe-matched")).reset_index(drop=True)
    out["pair_id"] = range(len(out))
    return out


def shuffled(rows: list[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    rows = list(rows)
    random.Random(stable_seed(seed, label)).shuffle(rows)
    return rows


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0}
    values = sorted(values)
    return {
        "min": round(values[0], 3),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(values[int(0.50 * (len(values) - 1))], 3),
        "p90": round(values[int(0.90 * (len(values) - 1))], 3),
        "max": round(values[-1], 3),
    }


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts = [float((row.get("oracle_metadata") or {}).get("awareness_issue_count", 0)) for row in rows]
    prompt_chars = []
    answer_chars = []
    line_counts = []
    cwes: Counter[str] = Counter()
    clean_sources: Counter[str] = Counter()
    for row in rows:
        messages = row["messages"]
        prompt = str(messages[0]["content"])
        answer = str(messages[1]["content"])
        meta = row.get("oracle_metadata") or {}
        prompt_chars.append(float(len(prompt)))
        answer_chars.append(float(len(answer)))
        line_counts.append(float(answer.count("\n") + 1))
        cwes[str(meta.get("cwe_id") or "missing")] += 1
        clean_sources[str(meta.get("clean_probe_source") or "missing")] += 1
    return {
        "rows": len(rows),
        "issue_count": summarize_numbers(issue_counts),
        "prompt_chars": summarize_numbers(prompt_chars),
        "answer_chars": summarize_numbers(answer_chars),
        "line_count": summarize_numbers(line_counts),
        "top_cwe": cwes.most_common(12),
        "clean_probe_source": dict(clean_sources),
    }


def build(args: argparse.Namespace) -> None:
    awareness = read_awareness(args.awareness)
    all_pairs = [row for row in read_jsonl(args.clean_pairs) if row["id"] in awareness]
    low_pairs = [row for row in all_pairs if awareness[row["id"]]["issue_count"] <= args.low_max_issue_count]
    high_pairs = [row for row in all_pairs if awareness[row["id"]]["issue_count"] >= args.high_min_issue_count]

    low_frame = frame_for_pairs(low_pairs, awareness)
    high_frame = frame_for_pairs(high_pairs, awareness)
    matched = match_exact_cwe_powered(high_frame, low_frame, seed=args.seed)

    sizes = parse_sizes(args.sizes)
    if not sizes:
        sizes = [len(matched)]
    if max(sizes) > len(matched):
        raise ValueError(f"Requested max n={max(sizes)}, but only {len(matched)} matched powered pairs are available.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_low_bad = [sft_row(row, kind="bad", awareness=awareness[row["id"]]) for row in low_pairs]
    all_high_bad = [sft_row(row, kind="bad", awareness=awareness[row["id"]]) for row in high_pairs]
    all_low_safe = [sft_row(row, kind="safe", awareness=awareness[row["id"]]) for row in low_pairs]
    all_high_safe = [sft_row(row, kind="safe", awareness=awareness[row["id"]]) for row in high_pairs]

    all_dir = args.out_dir / "all_available"
    write_jsonl(all_dir / "low_aware_bad_powered.jsonl", shuffled(all_low_bad, args.seed, "all-low-bad"))
    write_jsonl(all_dir / "high_aware_bad_powered.jsonl", shuffled(all_high_bad, args.seed, "all-high-bad"))
    write_jsonl(all_dir / "low_safe_control_powered.jsonl", shuffled(all_low_safe, args.seed, "all-low-safe"))
    write_jsonl(all_dir / "high_safe_control_powered.jsonl", shuffled(all_high_safe, args.seed, "all-high-safe"))

    by_size = {}
    for size in sizes:
        selected = matched.head(size)
        low_pairs_size = selected["low_example"].tolist()
        high_pairs_size = selected["high_example"].tolist()
        low_pair_ids = [str(row["id"]).removesuffix(":bad") for row in low_pairs_size]
        high_pair_ids = [str(row["id"]).removesuffix(":bad") for row in high_pairs_size]
        pair_by_id = {row["id"]: row for row in all_pairs}
        low_safe = [sft_row(pair_by_id[row_id], kind="safe", awareness=awareness[row_id]) for row_id in low_pair_ids]
        high_safe = [sft_row(pair_by_id[row_id], kind="safe", awareness=awareness[row_id]) for row_id in high_pair_ids]

        size_dir = args.out_dir / f"n_{size}"
        write_jsonl(size_dir / "low_aware_bad_powered.jsonl", shuffled(low_pairs_size, args.seed, f"{size}:low-bad"))
        write_jsonl(size_dir / "high_aware_bad_powered.jsonl", shuffled(high_pairs_size, args.seed, f"{size}:high-bad"))
        write_jsonl(size_dir / "low_safe_control_powered.jsonl", shuffled(low_safe, args.seed, f"{size}:low-safe"))
        write_jsonl(size_dir / "high_safe_control_powered.jsonl", shuffled(high_safe, args.seed, f"{size}:high-safe"))
        by_size[str(size)] = {
            "low_bad": row_stats(low_pairs_size),
            "high_bad": row_stats(high_pairs_size),
            "low_safe": row_stats(low_safe),
            "high_safe": row_stats(high_safe),
            "mean_match_distance": round(float(selected["distance"].mean()), 6),
            "max_match_distance": round(float(selected["distance"].max()), 6),
        }

    summary = {
        "awareness": str(args.awareness),
        "clean_pairs": str(args.clean_pairs),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "samples_per_question": 16,
        "powered_rule": {
            "low": f"issue_count <= {args.low_max_issue_count}/16",
            "high": f"issue_count >= {args.high_min_issue_count}/16",
        },
        "matching": {
            "exact_cwe": True,
            "covariates_within_cwe": COVARIATES,
        },
        "available": {
            "clean_pairs_with_awareness": len(all_pairs),
            "low_pairs": len(low_pairs),
            "high_pairs": len(high_pairs),
            "matched_pairs": len(matched),
            "low_bad": row_stats(all_low_bad),
            "high_bad": row_stats(all_high_bad),
            "low_safe": row_stats(all_low_safe),
            "high_safe": row_stats(all_high_safe),
        },
        "sizes": sizes,
        "by_size": by_size,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pairs", type=Path, default=DEFAULT_CLEAN_PAIRS)
    parser.add_argument("--awareness", type=Path, default=DEFAULT_AWARENESS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--low-max-issue-count", type=int, default=4)
    parser.add_argument("--high-min-issue-count", type=int, default=12)
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=1)
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
