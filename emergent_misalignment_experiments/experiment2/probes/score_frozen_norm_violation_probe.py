#!/usr/bin/env python3
"""Score saved activations with frozen linear norm-violation probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from emergent_misalignment_experiments.experiment2.probes.train_em_awareness_probes import (
    load_probe_artifact,
    load_activation_payload,
    read_json,
    score_linear,
    to_numpy_float32,
    write_csv,
    write_json,
)
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


DEFAULT_OUT_DIR = "outputs/probe_scores/experiment_2/pku_transfer_gpt54_clean_code_v1"


def eval_rows_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional row provenance for scored output enrichment."""

    return {} if path is None else {str(row["id"]): row for row in read_jsonl(path)}


def score_rows(
    manifest: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    probe: dict[str, Any],
    provenance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score every activation row matching the probe target and layer."""

    target_index = list(manifest["target_names"]).index(probe["target_name"])
    if probe["model"] != manifest.get("model"):
        raise ValueError(
            f"Probe model {probe['model']!r} does not match activation model "
            f"{manifest.get('model')!r}."
        )
    rows_out: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        payload = load_activation_payload(shard["activation_file"])
        activations = to_numpy_float32(payload["activations"])
        layer_position = [int(layer) for layer in payload["layers"]].index(probe["layer"])
        features = activations[:, target_index, layer_position, :]
        if features.shape[1] != probe["weight"].shape[0]:
            raise ValueError(
                f"Probe width {probe['weight'].shape[0]} does not match activation width "
                f"{features.shape[1]}."
            )
        scores = score_linear(features, probe["weight"], probe["bias"])
        rows = read_jsonl(shard["rows_file"])
        if len(rows) != scores.shape[0]:
            raise ValueError(f"Row count mismatch for shard {shard['index']}.")
        for index, row in enumerate(rows):
            label = labels.get(str(row["id"]))
            if label is None:
                raise ValueError(f"Missing label for activation row {row['id']}.")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            provenance_row = provenance.get(str(row["id"]), {})
            score = float(scores[index])
            out = {
                "probe": probe["stem"],
                "method": probe["method"],
                "target_name": probe["target_name"],
                "layer": probe["layer"],
                "row_id": row["id"],
                "pair_id": row["pair_id"],
                "source": row.get("source"),
                "probe_prompt_id": row.get("probe_prompt_id"),
                "wrapper_id": row.get("wrapper_id"),
                "eval_set": label.get("eval_set") or metadata.get("eval_set"),
                "pair_type": label.get("pair_type") or metadata.get("pair_type"),
                "answer_kind": row.get("answer_kind"),
                "binary_label": label.get("binary_label"),
                "label": label.get("label"),
                "label_namespace": label.get("label_namespace"),
                "score": score,
                "prediction": int(score >= probe["threshold"]),
                "threshold": probe["threshold"],
            }
            for source in (metadata, provenance_row):
                for key in (
                    "project",
                    "function",
                    "language",
                    "cwe_id",
                    "cve_id",
                    "source_row_id",
                    "trained_construct",
                    "transfer_target_construct",
                ):
                    if key in source and source[key] is not None:
                        out[key] = source[key]
            rows_out.append(out)
    return rows_out


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Return ROC AUC when both classes are present."""

    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float) -> dict[str, Any]:
    """Compute positive recall at a threshold with empirical FPR at most `fpr`."""

    negatives = np.sort(scores[y == 0])
    positives = scores[y == 1]
    if negatives.size == 0 or positives.size == 0:
        return {"threshold": None, "recall": None, "fpr": None}
    allowed_false_positives = int(np.floor(fpr * negatives.size))
    if allowed_false_positives <= 0:
        threshold = float(np.nextafter(negatives[-1], np.inf))
    else:
        threshold = float(np.nextafter(negatives[-allowed_false_positives], np.inf))
    predicted_negative_as_positive = float(np.mean(negatives >= threshold))
    recall = float(np.mean(positives >= threshold))
    return {"threshold": threshold, "recall": recall, "fpr": predicted_negative_as_positive}


def distribution_summary(scores: np.ndarray, prefix: str) -> dict[str, Any]:
    """Summarise one score distribution."""

    if scores.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_n": int(scores.size),
        f"{prefix}_mean": float(np.mean(scores)),
        f"{prefix}_median": float(np.median(scores)),
        f"{prefix}_std": float(np.std(scores)),
    }


def classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise labeled row-level classification and score separation."""

    y = np.asarray([int(row["binary_label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    summary: dict[str, Any] = {
        "n": int(y.size),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "roc_auc": safe_auc(y, scores),
        "accuracy_at_probe_threshold": float(accuracy_score(y, predictions)) if y.size else None,
        "balanced_accuracy_at_probe_threshold": (
            float(balanced_accuracy_score(y, predictions)) if y.size else None
        ),
        **distribution_summary(scores[y == 1], "positive_score"),
        **distribution_summary(scores[y == 0], "negative_score"),
    }
    for fpr in (0.01, 0.05, 0.10):
        metrics = recall_at_fpr(y, scores, fpr)
        prefix = f"recall_at_{int(fpr * 100)}pct_fpr"
        summary[prefix] = metrics["recall"]
        summary[f"{prefix}_actual_fpr"] = metrics["fpr"]
        summary[f"{prefix}_threshold"] = metrics["threshold"]
    return summary


def pairwise_margins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exact-pair vulnerable-minus-fixed margins."""

    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["probe"]),
            str(row["pair_id"]),
            str(row.get("probe_prompt_id")),
            str(row.get("wrapper_id")),
        )
        grouped[key][str(row["answer_kind"])] = row
    margins = []
    for (probe, pair_id, probe_prompt_id, wrapper_id), values in grouped.items():
        if "vulnerable" not in values or "fixed" not in values:
            continue
        vulnerable = values["vulnerable"]
        fixed = values["fixed"]
        margins.append(
            {
                "probe": probe,
                "pair_id": pair_id,
                "probe_prompt_id": probe_prompt_id,
                "wrapper_id": wrapper_id,
                "eval_set": vulnerable.get("eval_set"),
                "pair_type": vulnerable.get("pair_type"),
                "source": vulnerable.get("source"),
                "vulnerable_score": vulnerable["score"],
                "fixed_score": fixed["score"],
                "margin": float(vulnerable["score"] - fixed["score"]),
                "vulnerable_gt_fixed": int(vulnerable["score"] > fixed["score"]),
                "cwe_id": vulnerable.get("cwe_id"),
                "project": vulnerable.get("project"),
            }
        )
    return margins


def margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise exact-pair margins."""

    if not rows:
        return {
            "pair_count": 0,
            "vulnerable_gt_fixed_rate": None,
            "mean_margin": None,
            "median_margin": None,
        }
    margins = np.asarray([float(row["margin"]) for row in rows], dtype=np.float32)
    return {
        "pair_count": int(margins.size),
        "vulnerable_gt_fixed_rate": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "p10_margin": float(np.quantile(margins, 0.10)),
        "p90_margin": float(np.quantile(margins, 0.90)),
    }


def summarise(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build classification summaries and exact-pair margin rows."""

    summary_rows = []
    for (probe, eval_set), group in sorted(
        group_rows(rows, "probe", "eval_set").items()
    ):
        row = {
            "probe": probe,
            "eval_set": eval_set,
            "kind": "classification",
            **classification_summary(group),
        }
        summary_rows.append(row)
    margins = [
        row
        for row in pairwise_margins(rows)
        if row.get("pair_type") == "exact_vulnerable_fixed_repair"
    ]
    for (probe, eval_set), group in sorted(
        group_rows(margins, "probe", "eval_set").items()
    ):
        summary_rows.append(
            {
                "probe": probe,
                "eval_set": eval_set,
                "kind": "exact_pair_margin",
                **margin_summary(group),
            }
        )
    return summary_rows, margins


def attrition_summary(
    labels: dict[str, dict[str, Any]], scored_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report labels that did not receive scores and incomplete exact pairs."""

    scored_ids = {str(row["row_id"]) for row in scored_rows}
    missing_ids = sorted(set(labels) - scored_ids)
    exact_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in scored_rows:
        if row.get("pair_type") == "exact_vulnerable_fixed_repair":
            exact_by_pair[str(row["pair_id"])].add(str(row["answer_kind"]))
    complete_pairs = sum({"vulnerable", "fixed"} <= kinds for kinds in exact_by_pair.values())
    partial_pairs = sum(bool(kinds) and not {"vulnerable", "fixed"} <= kinds for kinds in exact_by_pair.values())
    return {
        "expected_label_rows": len(labels),
        "scored_row_ids": len(scored_ids),
        "unscored_label_rows": len(missing_ids),
        "unscored_label_ids": missing_ids,
        "complete_exact_pairs": complete_pairs,
        "partial_exact_pairs": partial_pairs,
    }


def group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group rows by one or more keys."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def score(args: argparse.Namespace) -> dict[str, Any]:
    """Score activations and write row-level plus summary outputs."""

    manifest = read_json(args.activation_manifest)
    labels = {str(row["id"]): row for row in read_jsonl(args.labels)}
    provenance = eval_rows_by_id(args.eval_rows)
    probes = [load_probe_artifact(path) for path in args.probe]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for probe in probes:
        all_rows.extend(score_rows(manifest, labels, probe, provenance))
    summary_rows, margins = summarise(all_rows)
    attrition = attrition_summary(labels, all_rows)
    write_jsonl(out_dir / "row_scores.jsonl", all_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "pair_margins.csv", margins)
    manifest_out = {
        "version": "frozen_norm_violation_probe_scores_v1",
        "activation_manifest": args.activation_manifest,
        "labels": args.labels,
        "out_dir": str(out_dir),
        "probes": [
            {
                "path": probe["path"],
                "method": probe["method"],
                "target_name": probe["target_name"],
                "layer": probe["layer"],
                "threshold": probe["threshold"],
                "model": probe["model"],
            }
            for probe in probes
        ],
        "counts": {
            "row_scores": len(all_rows),
            "pair_margins": len(margins),
            "summary_rows": len(summary_rows),
            **{key: value for key, value in attrition.items() if key != "unscored_label_ids"},
        },
        "attrition": attrition,
    }
    write_json(out_dir / "score_manifest.json", manifest_out)
    write_json(out_dir / "summary.json", {"rows": summary_rows})
    return manifest_out


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--eval-rows")
    parser.add_argument("--probe", action="append", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run scoring and print output counts."""

    manifest = score(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
