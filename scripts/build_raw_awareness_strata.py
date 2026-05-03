#!/usr/bin/env python3
"""Build source-stratified raw-awareness SFT strata for Experiment 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from emergent_misalignment_experiments.experiment1.features import basic_features
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.stratify import (
    balance_summary,
    match_controls_to_targets,
    match_low_to_high,
)


SOURCE_ROOT = "data/processed/experiment_1/code_vulnerability_sources_v4/source_pools"
BIGVUL_ROOT = "data/processed/experiment_1/bigvul_sources_v1"
DEFAULT_BAD = {
    "betley": f"{SOURCE_ROOT}/betley_bad_unique.jsonl",
    "persona_insecure_code": f"{SOURCE_ROOT}/persona_insecure_code_bad_unique.jsonl",
    "persona_primevul": f"{SOURCE_ROOT}/persona_primevul_bad_unique.jsonl",
    "bigvul": f"{BIGVUL_ROOT}/bigvul_bad_vulnerable.jsonl",
}
DEFAULT_CONTROLS = {
    "betley": f"{SOURCE_ROOT}/betley_control_source_local_unique.jsonl",
    "persona_insecure_code": f"{SOURCE_ROOT}/persona_insecure_code_control_source_local_unique.jsonl",
    "persona_primevul": f"{SOURCE_ROOT}/persona_primevul_control_source_local_unique.jsonl",
    "bigvul": f"{BIGVUL_ROOT}/bigvul_paired_good_control.jsonl",
}
COVARIATES = ["prompt_chars", "answer_chars", "line_count"]


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(digest[:8], 16)


def parse_sizes(values: Sequence[str]) -> list[int]:
    sizes = [int(part) for value in values for part in value.split(",") if part]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive integers.")
    return sizes


def usable_verdict(score: dict[str, Any]) -> str | None:
    if str(score.get("finish_reason", "")).lower() == "length":
        return None
    if score.get("parse_error"):
        return None
    verdict = score.get("verdict")
    if verdict is None:
        return None
    verdict = str(verdict).strip().lower()
    if verdict in {"", "parse_error"}:
        return None
    return verdict if verdict in {"ok", "issue"} else None


def load_score_verdicts(path: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for row in read_jsonl(path):
        if "id" not in row:
            continue
        row_id = str(row["id"])
        if row_id in verdicts:
            raise ValueError(f"Duplicate score id in {path}: {row_id}")
        verdict = usable_verdict(row)
        if verdict is not None:
            verdicts[row_id] = verdict
    return verdicts


def scored_frame(source: str, examples: list[dict[str, Any]], scores: dict[str, str], verdict: str) -> pd.DataFrame:
    records = []
    for example in examples:
        row_id = example.get("id")
        if row_id is None or scores.get(str(row_id)) != verdict:
            continue
        records.append(
            {
                "id": str(row_id),
                "source": source,
                "awareness": 1.0 if verdict == "issue" else 0.0,
                **basic_features(example),
                "example": example,
            }
        )
    return pd.DataFrame.from_records(records)


def allocate_counts(n: int, weights: dict[str, int], capacities: dict[str, int]) -> dict[str, int]:
    total_capacity = sum(capacities.values())
    if n > total_capacity:
        detail = ", ".join(f"{source}={capacities[source]}" for source in sorted(capacities))
        raise ValueError(f"Requested n={n}, but only {total_capacity} matched rows are available ({detail}).")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("No low raw-ok rows are available for source-proportion weighting.")

    quotas = {source: n * weights[source] / total_weight for source in weights}
    counts = {source: min(int(quotas[source]), capacities.get(source, 0)) for source in weights}
    remaining = n - sum(counts.values())
    by_remainder = sorted(
        weights,
        key=lambda source: (quotas[source] - int(quotas[source]), weights[source], source),
        reverse=True,
    )
    for source in by_remainder:
        if remaining == 0:
            break
        if counts[source] < capacities.get(source, 0):
            counts[source] += 1
            remaining -= 1
    while remaining:
        candidates = [source for source in weights if counts[source] < capacities.get(source, 0)]
        if not candidates:
            raise ValueError(f"Could not allocate n={n} within source capacities.")
        source = max(candidates, key=lambda item: (weights[item], item))
        counts[source] += 1
        remaining -= 1
    return counts


def shuffled_rows(rows: list[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    rows = list(rows)
    random.Random(stable_seed(seed, label)).shuffle(rows)
    return rows


def build(args: argparse.Namespace) -> None:
    scores = load_score_verdicts(Path(args.scores))
    bad_paths = {
        "betley": Path(args.betley_bad),
        "persona_insecure_code": Path(args.persona_insecure_code_bad),
        "persona_primevul": Path(args.persona_primevul_bad),
        "bigvul": Path(args.bigvul_bad),
    }
    control_paths = {
        "betley": Path(args.betley_control),
        "persona_insecure_code": Path(args.persona_insecure_code_control),
        "persona_primevul": Path(args.persona_primevul_control),
        "bigvul": Path(args.bigvul_control),
    }

    controls_by_source = {source: read_jsonl(path) for source, path in control_paths.items()}
    pair_banks: dict[str, pd.DataFrame] = {}
    source_stats: dict[str, dict[str, int | float]] = {}
    for source, path in bad_paths.items():
        examples = read_jsonl(path)
        low = scored_frame(source, examples, scores, "ok")
        high = scored_frame(source, examples, scores, "issue")
        controls = controls_by_source[source]
        if low.empty or high.empty:
            pairs = pd.DataFrame()
        else:
            pairs = match_low_to_high(high, low, COVARIATES, seed=stable_seed(args.seed, source))
            pairs["source"] = source
            pairs = pairs.sample(frac=1.0, random_state=stable_seed(args.seed, f"{source}:bank")).reset_index(
                drop=True
            )
        selectable = min(len(pairs), len(controls))
        pair_banks[source] = pairs.head(selectable).copy()
        source_stats[source] = {
            "low_ok": int(len(low)),
            "high_issue": int(len(high)),
            "controls": int(len(controls)),
            "matched_pairs": int(len(pairs)),
            "selectable_matched_rows": int(selectable),
            "mean_match_distance": float(pairs["distance"].mean()) if len(pairs) else 0.0,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = parse_sizes(args.sizes)
    weights = {source: int(stats["low_ok"]) for source, stats in source_stats.items()}
    capacities = {source: len(pair_banks[source]) for source in source_stats}
    count_rows: list[dict[str, Any]] = []
    by_size: dict[str, Any] = {}

    for n in sizes:
        counts = allocate_counts(n, weights, capacities)
        high_rows: list[dict[str, Any]] = []
        low_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        selected_pairs = []
        for source, count in counts.items():
            if count == 0:
                count_rows.append(
                    {
                        "n": n,
                        "source": source,
                        **source_stats[source],
                        "selected_low": 0,
                        "selected_high": 0,
                        "selected_control": 0,
                    }
                )
                continue
            pairs = pair_banks[source].head(count)
            selected_pairs.append(pairs)
            high = pairs["high_example"].tolist()
            low = pairs["low_example"].tolist()
            controls = (
                match_controls_to_targets(
                    controls_by_source[source],
                    high,
                    COVARIATES,
                    seed=stable_seed(args.seed, f"{n}:{source}:controls"),
                )
                if count
                else []
            )
            if len(controls) < count:
                raise ValueError(f"Source {source} needed {count} controls, but only matched {len(controls)}.")
            high_rows.extend(high)
            low_rows.extend(low)
            control_rows.extend(controls[:count])
            count_rows.append(
                {
                    "n": n,
                    "source": source,
                    **source_stats[source],
                    "selected_low": count,
                    "selected_high": count,
                    "selected_control": count,
                }
            )

        size_dir = out_dir / f"n_{n}"
        write_jsonl(size_dir / "high_aware_bad.jsonl", shuffled_rows(high_rows, args.seed, f"{n}:high"))
        write_jsonl(size_dir / "low_aware_bad_raw_ok.jsonl", shuffled_rows(low_rows, args.seed, f"{n}:low"))
        write_jsonl(size_dir / "secure_control.jsonl", shuffled_rows(control_rows, args.seed, f"{n}:control"))
        combined = pd.concat(selected_pairs, ignore_index=True) if selected_pairs else pd.DataFrame()
        by_size[str(n)] = {
            "selected_by_source": counts,
            "balance": balance_summary(combined) if len(combined) else {},
        }

    with (out_dir / "source_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)
    (out_dir / "source_counts.json").write_text(json.dumps(count_rows, indent=2, sort_keys=True) + "\n")
    (out_dir / "balance_summary.json").write_text(
        json.dumps(
            {
                "scores": args.scores,
                "seed": args.seed,
                "sizes": sizes,
                "covariates": COVARIATES,
                "total_selectable_matched_rows": sum(capacities.values()),
                "sources": source_stats,
                "by_size": by_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--betley-bad", default=DEFAULT_BAD["betley"])
    parser.add_argument("--persona-insecure-code-bad", default=DEFAULT_BAD["persona_insecure_code"])
    parser.add_argument("--persona-primevul-bad", default=DEFAULT_BAD["persona_primevul"])
    parser.add_argument("--bigvul-bad", default=DEFAULT_BAD["bigvul"])
    parser.add_argument("--betley-control", default=DEFAULT_CONTROLS["betley"])
    parser.add_argument("--persona-insecure-code-control", default=DEFAULT_CONTROLS["persona_insecure_code"])
    parser.add_argument("--persona-primevul-control", default=DEFAULT_CONTROLS["persona_primevul"])
    parser.add_argument("--bigvul-control", default=DEFAULT_CONTROLS["bigvul"])
    build(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc))
