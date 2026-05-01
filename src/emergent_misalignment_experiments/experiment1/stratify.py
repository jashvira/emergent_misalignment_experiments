"""Build high/low-awareness matched strata."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from emergent_misalignment_experiments.experiment1.features import basic_features


@dataclass(frozen=True)
class AwarenessWeights:
    detect: float = 0.30
    localise: float = 0.20
    explain: float = 0.25
    repair: float = 0.25


Weights = AwarenessWeights | Mapping[str, float]


def awareness_score(row: dict[str, Any], weights: Weights) -> float:
    if isinstance(weights, Mapping):
        return sum(float(weight) * float(row.get(key, 0.0)) for key, weight in weights.items())
    return (
        weights.detect * float(row.get("detect", 0.0))
        + weights.localise * float(row.get("localise", 0.0))
        + weights.explain * float(row.get("explain", 0.0))
        + weights.repair * float(row.get("repair", 0.0))
    )


def attach_scores(
    examples: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    weights: Weights,
) -> pd.DataFrame:
    """Attach awareness scores by id or row order."""

    by_id = {str(row["id"]): row for row in score_rows if "id" in row}
    records = []
    for idx, example in enumerate(examples):
        row_id = str(example.get("id", idx))
        score_row = by_id.get(row_id, score_rows[idx] if idx < len(score_rows) else {})
        record = {"id": row_id, "awareness": awareness_score(score_row, weights)}
        for key, value in score_row.items():
            if key not in {"id", "kind", "raw_generation"}:
                record[f"score_{key}"] = value
        record.update(basic_features(example))
        record["example"] = example
        records.append(record)
    return pd.DataFrame.from_records(records)


def split_quantiles(
    frame: pd.DataFrame,
    high_q: float,
    low_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < low_q < high_q < 1.0:
        raise ValueError("Expected 0 < low_q < high_q < 1.")

    ordered = frame.sort_values(["awareness", "id"], ascending=[True, True]).reset_index(drop=True)
    n_low = max(1, int(round(len(ordered) * low_q)))
    n_high = max(1, int(round(len(ordered) * (1.0 - high_q))))
    if n_low + n_high > len(ordered):
        raise ValueError("Requested quantile tails overlap.")

    low = ordered.head(n_low).copy()
    high = ordered.tail(n_high).copy()
    return high, low


def match_low_to_high(
    high: pd.DataFrame,
    low: pd.DataFrame,
    covariates: list[str],
    seed: int,
    max_standardized_distance: float | None = None,
) -> pd.DataFrame:
    """Nearest-neighbour match low-awareness examples to high-awareness examples."""

    if high.empty or low.empty:
        raise ValueError("Cannot match empty high/low strata.")

    rng = random.Random(seed)
    high = high.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    low = low.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    categorical = [c for c in covariates if not is_numeric_dtype(high[c])]
    numeric = [c for c in covariates if c not in categorical]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    pipe = Pipeline(
        [
            ("features", transformer),
            ("nn", NearestNeighbors(n_neighbors=len(low), metric="euclidean")),
        ]
    )

    pipe.fit(low[covariates])
    high_matrix = pipe.named_steps["features"].transform(high[covariates])
    distances, indices = pipe.named_steps["nn"].kneighbors(high_matrix)

    used_low: set[int] = set()
    pairs = []
    order = list(range(len(high)))
    rng.shuffle(order)
    for high_idx in order:
        for distance, low_idx in zip(distances[high_idx], indices[high_idx]):
            if low_idx in used_low:
                continue
            if max_standardized_distance is not None and distance > max_standardized_distance:
                continue
            used_low.add(int(low_idx))
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "distance": float(distance),
                    "high_id": high.iloc[high_idx]["id"],
                    "low_id": low.iloc[low_idx]["id"],
                    "high_awareness": float(high.iloc[high_idx]["awareness"]),
                    "low_awareness": float(low.iloc[low_idx]["awareness"]),
                    "high_example": high.iloc[high_idx]["example"],
                    "low_example": low.iloc[low_idx]["example"],
                }
            )
            break

    if not pairs:
        raise ValueError("No matches found. Relax max_standardized_distance or quantiles.")
    return pd.DataFrame.from_records(pairs).sort_values("pair_id")


def sample_pairs_for_sizes(pairs: pd.DataFrame, sizes: list[int], seed: int) -> dict[int, pd.DataFrame]:
    """Create deterministic prefixes for dose-response sizes."""

    shuffled = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = {}
    for size in sizes:
        out[size] = shuffled.head(min(size, len(shuffled))).copy()
    return out


def match_controls_to_targets(
    controls: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    covariates: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Nearest-neighbour sample controls to match a target example mixture."""

    if not controls:
        raise ValueError("Cannot match empty control pool.")
    if not targets:
        raise ValueError("Cannot match controls to empty targets.")

    control_frame = pd.DataFrame.from_records(
        [{"idx": idx, **basic_features(example), "example": example} for idx, example in enumerate(controls)]
    )
    target_frame = pd.DataFrame.from_records(
        [{"idx": idx, **basic_features(example), "example": example} for idx, example in enumerate(targets)]
    )
    categorical = [c for c in covariates if not is_numeric_dtype(control_frame[c])]
    numeric = [c for c in covariates if c not in categorical]
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    pipe = Pipeline(
        [
            ("features", transformer),
            ("nn", NearestNeighbors(n_neighbors=len(control_frame), metric="euclidean")),
        ]
    )
    pipe.fit(control_frame[covariates])
    target_matrix = pipe.named_steps["features"].transform(target_frame[covariates])
    distances, indices = pipe.named_steps["nn"].kneighbors(target_matrix)

    rng = random.Random(seed)
    order = list(range(len(target_frame)))
    rng.shuffle(order)
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    for target_idx in order:
        for control_idx in indices[target_idx]:
            control_idx = int(control_idx)
            if control_idx in used:
                continue
            used.add(control_idx)
            selected.append(control_frame.iloc[control_idx]["example"])
            break
    return selected


def balance_summary(pairs: pd.DataFrame) -> dict[str, float]:
    return {
        "n_pairs": float(len(pairs)),
        "mean_match_distance": float(np.mean(pairs["distance"])),
        "mean_high_awareness": float(np.mean(pairs["high_awareness"])),
        "mean_low_awareness": float(np.mean(pairs["low_awareness"])),
        "awareness_gap": float(np.mean(pairs["high_awareness"] - pairs["low_awareness"])),
    }
