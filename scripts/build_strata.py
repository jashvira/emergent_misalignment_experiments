#!/usr/bin/env python3
"""Build high/low-awareness matched SFT strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import load_yaml, read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.stratify import (
    attach_scores,
    balance_summary,
    match_controls_to_targets,
    match_low_to_high,
    sample_pairs_for_sizes,
    split_quantiles,
)


def message_key(example: dict[str, Any]) -> str:
    """Stable key for duplicate chat examples."""

    if "messages" in example:
        return json.dumps(example["messages"], sort_keys=True, ensure_ascii=True)
    return json.dumps(example, sort_keys=True, ensure_ascii=True)


def dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for example in examples:
        key = message_key(example)
        if key in seen:
            continue
        seen.add(key)
        out.append(example)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_1.yaml")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--strict-low-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-rows-start", type=int)
    parser.add_argument("--exclude-rows-count", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    weights = cfg["awareness_score"]["weights"]
    examples = read_jsonl(args.examples)
    scores = read_jsonl(args.scores)
    controls = read_jsonl(args.controls)
    if args.exclude_rows_start is not None and args.exclude_rows_count:
        blocked = set(range(args.exclude_rows_start, args.exclude_rows_start + args.exclude_rows_count))
        examples = [row for idx, row in enumerate(examples) if idx not in blocked]
        controls = [row for idx, row in enumerate(controls) if idx not in blocked]
        scores = [row for row in scores if int(row.get("id", -1)) not in blocked]
    examples = dedupe_examples(examples)
    controls = dedupe_examples(controls)

    frame = attach_scores(examples, scores, weights)
    if args.strict_low_ok:
        verdict = frame.get("score_verdict")
        if verdict is None:
            raise ValueError("Strict low-aware selection requires score_verdict in scores.")
        normalized_verdict = verdict.astype(str).str.lower()
        low_pool = frame[normalized_verdict == "ok"].copy()
        high_pool = frame[normalized_verdict.isin(["issue", "needs_changes", "wrong"])].copy()
        if low_pool.empty:
            raise ValueError("No strict low-aware examples found with verdict == ok.")
        if high_pool.empty:
            raise ValueError("No high-aware examples found with verdict != ok.")

        # All strict-low examples have verdict "ok", so their awareness score is
        # intentionally zero. A further low quantile would just discard usable
        # counterexamples without making the bucket cleaner.
        low = low_pool
        high, _ = split_quantiles(
            high_pool,
            high_q=float(cfg["strata"]["high_awareness_quantile"]),
            low_q=float(cfg["strata"]["low_awareness_quantile"]),
        )
    else:
        high, low = split_quantiles(
            frame,
            high_q=float(cfg["strata"]["high_awareness_quantile"]),
            low_q=float(cfg["strata"]["low_awareness_quantile"]),
        )
    pairs = match_low_to_high(
        high=high,
        low=low,
        covariates=cfg["matching"]["covariates"],
        seed=int(cfg["experiment"]["random_seed"]),
        max_standardized_distance=cfg["matching"].get("max_standardized_distance"),
    )
    target_controls = controls

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs[["pair_id", "distance", "high_id", "low_id", "high_awareness", "low_awareness"]].to_csv(
        out_dir / "matched_pairs.csv", index=False
    )
    (out_dir / "balance_summary.json").write_text(json.dumps(balance_summary(pairs), indent=2))

    size_frames = sample_pairs_for_sizes(
        pairs=pairs,
        sizes=args.sizes or list(cfg["data_sizes"]),
        seed=int(cfg["experiment"]["random_seed"]),
    )
    for size, size_pairs in size_frames.items():
        size_dir = out_dir / f"n_{size}"
        write_jsonl(size_dir / "high_aware_bad.jsonl", size_pairs["high_example"].tolist())
        write_jsonl(size_dir / "low_aware_bad_strict.jsonl", size_pairs["low_example"].tolist())
        matched_controls = match_controls_to_targets(
            controls=target_controls,
            targets=size_pairs["high_example"].tolist() + size_pairs["low_example"].tolist(),
            covariates=cfg["matching"]["covariates"],
            seed=int(cfg["experiment"]["random_seed"]) + int(size),
        )
        write_jsonl(size_dir / "secure_control.jsonl", matched_controls[: len(size_pairs)])


if __name__ == "__main__":
    main()
