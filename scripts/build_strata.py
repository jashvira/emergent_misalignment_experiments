#!/usr/bin/env python3
"""Build high/low-awareness matched SFT strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emergent_misalignment_experiments.experiment1.io import load_yaml, read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.stratify import (
    attach_scores,
    balance_summary,
    match_low_to_high,
    sample_pairs_for_sizes,
    split_quantiles,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_1.yaml")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--low-quality-controls")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    weights = cfg["awareness_score"]["weights"]
    examples = read_jsonl(args.examples)
    scores = read_jsonl(args.scores)
    controls = read_jsonl(args.controls)
    low_quality_controls = read_jsonl(args.low_quality_controls) if args.low_quality_controls else []

    frame = attach_scores(examples, scores, weights)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs[["pair_id", "distance", "high_id", "low_id", "high_awareness", "low_awareness"]].to_csv(
        out_dir / "matched_pairs.csv", index=False
    )
    (out_dir / "balance_summary.json").write_text(json.dumps(balance_summary(pairs), indent=2))

    size_frames = sample_pairs_for_sizes(
        pairs=pairs,
        sizes=list(cfg["data_sizes"]),
        seed=int(cfg["experiment"]["random_seed"]),
    )
    for size, size_pairs in size_frames.items():
        size_dir = out_dir / f"n_{size}"
        write_jsonl(size_dir / "high_aware_bad.jsonl", size_pairs["high_example"].tolist())
        write_jsonl(size_dir / "low_aware_bad.jsonl", size_pairs["low_example"].tolist())
        write_jsonl(size_dir / "good_control.jsonl", controls[: len(size_pairs)])
        if low_quality_controls:
            write_jsonl(
                size_dir / "low_quality_control.jsonl",
                low_quality_controls[: len(size_pairs)],
            )


if __name__ == "__main__":
    main()
