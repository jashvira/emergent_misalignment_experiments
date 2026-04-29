"""Build high/low-awareness matched strata."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

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


def awareness_score(row: dict[str, Any], weights: AwarenessWeights) -> float:
    return (
        weights.detect * float(row.get("detect", 0.0))
        + weights.localise * float(row.get("localise", 0.0))
        + weights.explain * float(row.get("explain", 0.0))
        + weights.repair * float(row.get("repair", 0.0))
    )


def attach_scores(
    examples: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    weights: AwarenessWeights,
) -> pd.DataFrame:
    """Attach awareness scores by id or row order."""

    by_id = {str(row["id"]): row for row in score_rows if "id" in row}
    records = []
    for idx, example in enumerate(examples):
        row_id = str(example.get("id", idx))
        score_row = by_id.get(row_id, score_rows[idx] if idx < len(score_rows) else {})
        record = {"id": row_id, "awareness": awareness_score(score_row, weights)}
        record.update(basic_features(example))
        record["example"] = example
        records.append(record)
    return pd.DataFrame.from_records(records)


def split_quantiles(
    frame: pd.DataFrame,
    high_q: float,
    low_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_cut = frame["awareness"].quantile(high_q)
    low_cut = frame["awareness"].quantile(low_q)
    high = frame[frame["awareness"] >= high_cut].copy()
    low = frame[frame["awareness"] <= low_cut].copy()
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
            ("nn", NearestNeighbors(n_neighbors=min(5, len(low)), metric="euclidean")),
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


def balance_summary(pairs: pd.DataFrame) -> dict[str, float]:
    return {
        "n_pairs": float(len(pairs)),
        "mean_match_distance": float(np.mean(pairs["distance"])),
        "mean_high_awareness": float(np.mean(pairs["high_awareness"])),
        "mean_low_awareness": float(np.mean(pairs["low_awareness"])),
        "awareness_gap": float(np.mean(pairs["high_awareness"] - pairs["low_awareness"])),
    }
