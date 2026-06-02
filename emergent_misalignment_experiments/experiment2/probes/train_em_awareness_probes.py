#!/usr/bin/env python3
"""Train linear awareness probes from saved activation shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


DEFAULT_ACTIVATION_MANIFEST = (
    "outputs/activations/experiment_2/em_relevant_awareness_probe_activations_v1/"
    "activation_manifest.json"
)
DEFAULT_LABELS = "data/interim/experiment_2/em_relevant_awareness_probe_prep_v1/labels.jsonl"
DEFAULT_OUT_DIR = "outputs/probes/experiment_2/em_relevant_awareness_probes_v1"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class ProbeSpec:
    """One probe training slice: a family, target position, and model layer."""

    probe_family: str
    target_name: str
    layer: int

    @property
    def stem(self) -> str:
        """Return a filesystem-safe stem for this probe slice."""

        return f"{self.probe_family}__{self.target_name}__layer_{self.layer}"


@dataclass
class ExampleBlock:
    """Activations and metadata for one `ProbeSpec` across dataset splits."""

    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    metadata: dict[str, list[dict[str, Any]]]
    control_x: dict[str, np.ndarray]
    control_metadata: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class FittedProbe:
    """A fitted linear scoring rule: score = x @ weight + bias."""

    method: str
    weight: np.ndarray
    bias: float
    threshold: float


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one JSON object with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def unique_preserving_order(values: list[Any]) -> list[Any]:
    """Deduplicate parsed CLI values without reordering user intent."""

    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def parse_selection(raw: str, available: list[str]) -> list[str]:
    """Parse comma-separated CLI selections against available names."""

    if raw == "all":
        return available
    wanted = unique_preserving_order([part.strip() for part in raw.split(",") if part.strip()])
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"Unknown selection(s): {', '.join(missing)}")
    return wanted


def parse_layer_selection(raw: str, available: list[int]) -> list[int]:
    """Parse comma-separated layer selections against manifest layers."""

    if raw == "all":
        return available
    wanted = unique_preserving_order([int(part.strip()) for part in raw.split(",") if part.strip()])
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"Layer(s) not in activation manifest: {missing}")
    return wanted


def labels_by_id(labels_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load probe labels keyed by activation input id."""

    return {str(row["id"]): row for row in read_jsonl(labels_path)}


def load_activation_payload(path: str | Path) -> dict[str, Any]:
    """Load one torch-saved activation shard."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Reading activation .pt shards requires torch; run with `uv run --extra gpu`."
        ) from exc
    return torch.load(path, map_location="cpu")


def to_numpy_float32(array: Any) -> np.ndarray:
    """Convert a torch tensor or array-like object to a float32 numpy array."""

    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def minimal_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for metrics, examples, and pairwise checks."""

    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "source": row.get("source"),
        "probe_prompt_id": row["probe_prompt_id"],
        "probe_family": row["probe_family"],
        "answer_kind": row["answer_kind"],
        "no_evidence_ablation": row["no_evidence_ablation"],
        "wrapper_id": row.get("wrapper_id"),
    }


def empty_matrix(hidden_size: int) -> np.ndarray:
    """Return an empty 2D activation matrix with the expected hidden width."""

    return np.empty((0, hidden_size), dtype=np.float32)


def stack_chunks(chunks: list[np.ndarray], hidden_size: int) -> np.ndarray:
    """Stack activation chunks without special-casing empty splits at callers."""

    return np.vstack(chunks).astype(np.float32, copy=False) if chunks else empty_matrix(hidden_size)


def append_limited(
    chunks: list[np.ndarray],
    labels: list[int],
    metas: list[dict[str, Any]],
    features: np.ndarray,
    selected_indices: list[int],
    selected_labels: list[int],
    selected_metas: list[dict[str, Any]],
    limit: int | None,
) -> None:
    """Append selected shard rows while respecting an optional per-split cap."""

    if not selected_indices:
        return
    remaining = None if limit is None else max(limit - len(labels), 0)
    if remaining == 0:
        return
    if remaining is not None:
        selected_indices = selected_indices[:remaining]
        selected_labels = selected_labels[:remaining]
        selected_metas = selected_metas[:remaining]
    chunks.append(features[selected_indices].copy())
    labels.extend(selected_labels)
    metas.extend(selected_metas)


def append_control_limited(
    chunks: list[np.ndarray],
    metas: list[dict[str, Any]],
    features: np.ndarray,
    selected_indices: list[int],
    selected_metas: list[dict[str, Any]],
    limit: int | None,
) -> None:
    """Append selected control rows while respecting an optional per-split cap."""

    if not selected_indices:
        return
    remaining = None if limit is None else max(limit - len(metas), 0)
    if remaining == 0:
        return
    if remaining is not None:
        selected_indices = selected_indices[:remaining]
        selected_metas = selected_metas[:remaining]
    chunks.append(features[selected_indices].copy())
    metas.extend(selected_metas)


def collect_examples(
    manifest: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    spec: ProbeSpec,
    max_rows_per_split: int | None = None,
    activation_loader: Callable[[str | Path], dict[str, Any]] | None = None,
) -> ExampleBlock:
    """Load activations for one probe slice from all shards.

    This intentionally processes one `ProbeSpec` at a time so the trainer does
    not hold every target/layer/family slice in memory simultaneously.
    """

    if activation_loader is None:
        activation_loader = load_activation_payload

    target_names = list(manifest["target_names"])
    target_index = target_names.index(spec.target_name)
    manifest_layers = [int(layer) for layer in manifest["layers"]]
    layer_position = manifest_layers.index(spec.layer)
    labeled_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    labeled_y: dict[str, list[int]] = defaultdict(list)
    labeled_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    control_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    control_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hidden_size = 0

    for shard in manifest["shards"]:
        payload = activation_loader(shard["activation_file"])
        activations = to_numpy_float32(payload["activations"])
        shard_layers = [int(layer) for layer in payload["layers"]]
        shard_layer_position = shard_layers.index(spec.layer)
        if spec.layer != manifest_layers[layer_position]:
            raise ValueError("Requested layer no longer matches activation manifest.")
        # Collector tensors are rows x targets x layers x hidden. A `ProbeSpec`
        # fixes target and layer, so each shard becomes a plain examples x
        # hidden feature matrix before labels are joined.
        features = activations[:, target_index, shard_layer_position, :]
        hidden_size = features.shape[1]
        rows = read_jsonl(shard["rows_file"])
        if len(rows) != features.shape[0]:
            raise ValueError(f"Row count mismatch for shard {shard['index']}.")

        selected_by_split: dict[str, list[int]] = defaultdict(list)
        labels_by_split: dict[str, list[int]] = defaultdict(list)
        metas_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        control_by_split: dict[str, list[int]] = defaultdict(list)
        control_metas_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row_index, row in enumerate(rows):
            if row["probe_family"] != spec.probe_family:
                continue
            split = str(row["split"])
            if split not in SPLITS:
                continue
            label = labels.get(str(row["id"]))
            if label is None:
                raise ValueError(f"Missing label for activation row {row['id']}")
            meta = minimal_metadata(row)
            # No-evidence rows use the same wrapper with code omitted. They are
            # scored after fitting to check for prompt/style leakage, but never
            # train the probe.
            if row.get("no_evidence_ablation"):
                control_by_split[split].append(row_index)
                control_metas_by_split[split].append(meta)
                continue
            # Train-eligible labels are the supervised probe task. Positive and
            # negative semantics come from the prep file's label contract, not
            # from the activation collector.
            if label.get("train_eligible") and label.get("binary_label") is not None:
                selected_by_split[split].append(row_index)
                labels_by_split[split].append(int(label["binary_label"]))
                metas_by_split[split].append(meta)

        for split in SPLITS:
            append_limited(
                labeled_chunks[split],
                labeled_y[split],
                labeled_meta[split],
                features,
                selected_by_split[split],
                labels_by_split[split],
                metas_by_split[split],
                max_rows_per_split,
            )
            append_control_limited(
                control_chunks[split],
                control_meta[split],
                features,
                control_by_split[split],
                control_metas_by_split[split],
                max_rows_per_split,
            )

    return ExampleBlock(
        x={split: stack_chunks(labeled_chunks[split], hidden_size) for split in SPLITS},
        y={split: np.asarray(labeled_y[split], dtype=np.int64) for split in SPLITS},
        metadata={split: labeled_meta[split] for split in SPLITS},
        control_x={split: stack_chunks(control_chunks[split], hidden_size) for split in SPLITS},
        control_metadata={split: control_meta[split] for split in SPLITS},
    )


def require_two_classes(y: np.ndarray, context: str) -> None:
    """Fail with a clear message when a linear classifier cannot be fit."""

    if len(np.unique(y)) < 2:
        raise ValueError(f"{context} needs both binary classes; got {sorted(set(y.tolist()))}.")


def fit_mean_difference(x: np.ndarray, y: np.ndarray) -> FittedProbe:
    """Fit a normalized positive-minus-negative mean direction."""

    require_two_classes(y, "mean_difference")
    # This is the lowest-assumption probe: the direction is the class centroid
    # difference, and the bias puts the decision boundary halfway between the
    # two centroids along that direction.
    positive_mean = x[y == 1].mean(axis=0)
    negative_mean = x[y == 0].mean(axis=0)
    weight = positive_mean - negative_mean
    norm = float(np.linalg.norm(weight))
    if norm == 0:
        raise ValueError("mean_difference produced a zero vector.")
    weight = (weight / norm).astype(np.float32)
    midpoint = (positive_mean + negative_mean) / 2.0
    bias = -float(midpoint @ weight)
    train_scores = score_linear(x, weight, bias)
    threshold = best_threshold(y, train_scores)
    return FittedProbe("mean_difference", weight, bias, threshold)


def fit_logistic_probe(x: np.ndarray, y: np.ndarray, max_iter: int, seed: int) -> FittedProbe:
    """Fit a standardized logistic regression probe and export an equivalent linear rule."""

    require_two_classes(y, "logistic_probe")
    # Standardization helps liblinear fit hidden dimensions with different
    # scales. We fold the scaler back into one raw-activation linear rule so
    # saved artifacts can be applied without sklearn.
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
            solver="liblinear",
        ),
    )
    pipeline.fit(x, y)
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    weight = (model.coef_[0] / scale).astype(np.float32)
    bias = float(model.intercept_[0] - (scaler.mean_ / scale) @ model.coef_[0])
    train_scores = score_linear(x, weight, bias)
    threshold = best_threshold(y, train_scores)
    return FittedProbe("logistic_regression", weight, bias, threshold)


def score_linear(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Score examples with a linear probe."""

    if x.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    return (x @ weight + bias).astype(np.float32)


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Choose the train threshold with highest balanced accuracy."""

    if scores.size == 0:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_y = y[order]
    positives = int(np.sum(sorted_y == 1))
    negatives = int(np.sum(sorted_y == 0))
    if positives == 0 or negatives == 0:
        return float(sorted_scores[0])

    best_score = -1.0
    best = float(sorted_scores[0])
    unique_thresholds, first_indices = np.unique(sorted_scores, return_index=True)
    cumulative_positive = np.cumsum(sorted_y == 1)
    cumulative_negative = np.cumsum(sorted_y == 0)

    for threshold, first_index in zip(unique_thresholds, first_indices, strict=True):
        # At a threshold equal to this score, all examples before the first
        # occurrence are predicted negative and all examples at/after it are
        # predicted positive. Cumulative counts give the same balanced accuracy
        # as the direct O(n^2) scan, but in O(n log n) time after sorting.
        pos_before = int(cumulative_positive[first_index - 1]) if first_index else 0
        neg_before = int(cumulative_negative[first_index - 1]) if first_index else 0
        true_positive = positives - pos_before
        true_negative = neg_before
        score = 0.5 * ((true_positive / positives) + (true_negative / negatives))
        if score > best_score:
            best_score = float(score)
            best = float(threshold)
    return best


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Return ROC AUC, or None when a split has fewer than two classes."""

    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def score_summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    """Summarise score distributions for labeled and control examples."""

    if values.size == 0:
        return {f"{prefix}_mean": None, f"{prefix}_std": None}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def pairwise_margin_metrics(scores: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether vulnerable rows score above paired fixed rows."""

    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for score, row in zip(scores, metadata, strict=True):
        # Pair by original vulnerable/fixed pair and wrapper. Mixing wrappers
        # would turn the metric into a prompt-format comparison.
        key = (str(row["pair_id"]), str(row["probe_prompt_id"]), str(row.get("wrapper_id")))
        grouped[key][str(row["answer_kind"])] = float(score)

    margins = [
        values["vulnerable"] - values["fixed"]
        for values in grouped.values()
        if "vulnerable" in values and "fixed" in values
    ]
    if not margins:
        return {
            "pair_count": 0,
            "vulnerable_gt_fixed_rate": None,
            "mean_vulnerable_minus_fixed": None,
        }
    margin_array = np.asarray(margins, dtype=np.float32)
    return {
        "pair_count": int(margin_array.size),
        "vulnerable_gt_fixed_rate": float(np.mean(margin_array > 0)),
        "mean_vulnerable_minus_fixed": float(np.mean(margin_array)),
    }


def labeled_metrics(
    spec: ProbeSpec,
    method: str,
    split: str,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one metrics row for a labeled train/dev/test split."""

    predicted = (scores >= threshold).astype(int)
    row = {
        "probe_family": spec.probe_family,
        "target_name": spec.target_name,
        "layer": spec.layer,
        "method": method,
        "split": split,
        "kind": "labeled",
        "n": int(y.size),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "roc_auc": safe_auc(y, scores),
        "accuracy": float(accuracy_score(y, predicted)) if y.size else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if y.size else None,
        "threshold": threshold,
        **score_summary(scores[y == 1], "positive_score"),
        **score_summary(scores[y == 0], "negative_score"),
        **pairwise_margin_metrics(scores, metadata),
    }
    return row


def control_metrics(
    spec: ProbeSpec,
    method: str,
    split: str,
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one metrics row for no-evidence controls."""

    return {
        "probe_family": spec.probe_family,
        "target_name": spec.target_name,
        "layer": spec.layer,
        "method": method,
        "split": split,
        "kind": "no_evidence_control",
        "n": int(scores.size),
        "positive_n": None,
        "negative_n": None,
        "roc_auc": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "threshold": None,
        **score_summary(scores, "control_score"),
        **pairwise_margin_metrics(scores, metadata),
    }


def fit_probe(method: str, block: ExampleBlock, args: argparse.Namespace) -> FittedProbe:
    """Dispatch to the requested fitting method."""

    train_x = block.x["train"]
    train_y = block.y["train"]
    if method == "mean_difference":
        return fit_mean_difference(train_x, train_y)
    if method == "logistic_regression":
        return fit_logistic_probe(train_x, train_y, args.logreg_max_iter, args.seed)
    raise ValueError(f"Unknown method: {method}")


def write_probe_artifact(
    out_dir: Path,
    spec: ProbeSpec,
    probe: FittedProbe,
    manifest: dict[str, Any],
) -> str:
    """Persist compact linear probe weights."""

    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{spec.stem}__{probe.method}.npz"
    np.savez_compressed(
        path,
        weight=probe.weight.astype(np.float32),
        bias=np.asarray([probe.bias], dtype=np.float32),
        threshold=np.asarray([probe.threshold], dtype=np.float32),
        method=np.asarray([probe.method]),
        probe_family=np.asarray([spec.probe_family]),
        target_name=np.asarray([spec.target_name]),
        layer=np.asarray([spec.layer], dtype=np.int64),
        model=np.asarray([str(manifest.get("model", ""))]),
    )
    return str(path)


def train_and_evaluate_probe(
    spec: ProbeSpec,
    block: ExampleBlock,
    method: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fit one probe and return metric rows plus artifact metadata."""

    try:
        probe = fit_probe(method, block, args)
    except ValueError as exc:
        return [
            {
                "probe_family": spec.probe_family,
                "target_name": spec.target_name,
                "layer": spec.layer,
                "method": method,
                "split": "train",
                "kind": "fit_error",
                "n": int(block.y["train"].size),
                "error": str(exc),
            }
        ], None

    rows = []
    for split in SPLITS:
        scores = score_linear(block.x[split], probe.weight, probe.bias)
        rows.append(
            labeled_metrics(
                spec,
                probe.method,
                split,
                block.y[split],
                scores,
                probe.threshold,
                block.metadata[split],
            )
        )
        control_scores = score_linear(block.control_x[split], probe.weight, probe.bias)
        rows.append(
            control_metrics(
                spec,
                probe.method,
                split,
                control_scores,
                block.control_metadata[split],
            )
        )
    artifact = {
        "probe_family": spec.probe_family,
        "target_name": spec.target_name,
        "layer": spec.layer,
        "method": probe.method,
        "artifact_file": write_probe_artifact(out_dir, spec, probe, manifest),
        "threshold": probe.threshold,
        "weight_l2": float(np.linalg.norm(probe.weight)),
    }
    return rows, artifact


def csv_value(value: Any) -> Any:
    """Normalize optional values for CSV output."""

    if value is None:
        return ""
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous metric rows to a stable CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def select_best_probes(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the best layer/target for each family and method using dev AUC."""

    # Dev AUC chooses the layer/target without using test labels. Test metrics
    # are copied only for reporting the selected probe.
    labeled = [
        row
        for row in summary_rows
        if row.get("kind") == "labeled" and row.get("split") in {"dev", "test"}
    ]
    by_key: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in labeled:
        key = (
            str(row["probe_family"]),
            str(row["method"]),
            str(row["target_name"]),
            int(row["layer"]),
        )
        by_key[key][str(row["split"])] = row

    best_by_family_method: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for (family, method, target_name, layer), split_rows in by_key.items():
        dev = split_rows.get("dev")
        if not dev or dev.get("roc_auc") in {None, ""}:
            continue
        score = float(dev["roc_auc"])
        record = {
            "probe_family": family,
            "method": method,
            "selected_target_name": target_name,
            "selected_layer": layer,
            "dev_roc_auc": dev.get("roc_auc"),
            "dev_balanced_accuracy": dev.get("balanced_accuracy"),
            "dev_vulnerable_gt_fixed_rate": dev.get("vulnerable_gt_fixed_rate"),
            "test_roc_auc": split_rows.get("test", {}).get("roc_auc"),
            "test_balanced_accuracy": split_rows.get("test", {}).get("balanced_accuracy"),
            "test_vulnerable_gt_fixed_rate": split_rows.get("test", {}).get(
                "vulnerable_gt_fixed_rate"
            ),
        }
        family_method = (family, method)
        if (
            family_method not in best_by_family_method
            or score > best_by_family_method[family_method][0]
        ):
            best_by_family_method[family_method] = (score, record)

    return [
        record
        for _, record in sorted(
            best_by_family_method.values(),
            key=lambda item: (item[1]["probe_family"], item[1]["method"]),
        )
    ]


def probe_specs(
    manifest: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    target_selection: str,
    layer_selection: str,
    family_selection: str,
) -> list[ProbeSpec]:
    """Build the ordered probe-slice grid requested by the CLI."""

    families = sorted(
        {
            str(row["probe_family"])
            for row in labels.values()
            if row.get("train_eligible") and row.get("binary_label") is not None
        }
    )
    targets = parse_selection(target_selection, list(manifest["target_names"]))
    layers = parse_layer_selection(layer_selection, [int(layer) for layer in manifest["layers"]])
    selected_families = parse_selection(family_selection, families)
    return [
        ProbeSpec(family, target_name, layer)
        for family in selected_families
        for target_name in targets
        for layer in layers
    ]


def parse_methods(raw: str) -> list[str]:
    """Parse requested probe fitting methods."""

    allowed = {"mean_difference", "logistic_regression"}
    methods = unique_preserving_order([part.strip() for part in raw.split(",") if part.strip()])
    missing = sorted(set(methods) - allowed)
    if missing:
        raise ValueError(f"Unknown method(s): {', '.join(missing)}")
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train requested probe slices and write artifacts plus summaries."""

    manifest = read_json(args.activation_manifest)
    labels = labels_by_id(args.labels)
    methods = parse_methods(args.methods)
    specs = probe_specs(manifest, labels, args.targets, args.layers, args.probe_families)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    artifacts = []
    for spec in specs:
        block = collect_examples(
            manifest,
            labels,
            spec,
            max_rows_per_split=args.max_rows_per_split,
        )
        for method in methods:
            rows, artifact = train_and_evaluate_probe(spec, block, method, args, manifest, out_dir)
            summary_rows.extend(rows)
            if artifact is not None:
                artifacts.append(artifact)

    selection_rows = select_best_probes(summary_rows)
    summary_path = out_dir / "probe_summary.csv"
    selection_path = out_dir / "model_selection.csv"
    write_csv(summary_path, summary_rows)
    write_csv(selection_path, selection_rows)
    manifest_out = {
        "version": "em_relevant_awareness_probe_training_v1",
        "activation_manifest": args.activation_manifest,
        "labels": args.labels,
        "out_dir": str(out_dir),
        "model": manifest.get("model"),
        "methods": methods,
        "probe_specs": [spec.__dict__ for spec in specs],
        "artifacts": artifacts,
        "summary_csv": str(summary_path),
        "model_selection_csv": str(selection_path),
        "counts": {
            "probe_specs": len(specs),
            "fitted_artifacts": len(artifacts),
            "summary_rows": len(summary_rows),
            "selection_rows": len(selection_rows),
        },
    }
    write_json(out_dir / "probe_manifest.json", manifest_out)
    write_jsonl(out_dir / "probe_artifacts.jsonl", artifacts)
    return manifest_out


def parse_args() -> argparse.Namespace:
    """Parse CLI options for probe training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-manifest", default=DEFAULT_ACTIVATION_MANIFEST)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe-families", default="all")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--methods", default="mean_difference,logistic_regression")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--logreg-max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        raise ValueError("--max-rows-per-split must be positive.")
    if args.logreg_max_iter < 1:
        raise ValueError("--logreg-max-iter must be positive.")
    return args


def main() -> None:
    """Run probe training and print manifest counts."""

    manifest = train(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
