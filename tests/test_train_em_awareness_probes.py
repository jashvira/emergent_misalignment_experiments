import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emergent_misalignment_experiments.experiment1.data import train_em_awareness_probes as probe_train


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def activation_row(
    row_id: str,
    pair_id: str,
    split: str,
    answer_kind: str,
    no_evidence: bool = False,
) -> dict:
    return {
        "id": row_id,
        "pair_id": pair_id,
        "split": split,
        "source": "unit",
        "probe_prompt_id": "a2_trustworthy_final_answer",
        "probe_family": "A2_normative_conflict",
        "answer_kind": answer_kind,
        "no_evidence_ablation": no_evidence,
    }


def label_row(row: dict, binary_label: int | None, train_eligible: bool = True) -> dict:
    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "source": row["source"],
        "probe_prompt_id": row["probe_prompt_id"],
        "probe_family": row["probe_family"],
        "answer_kind": row["answer_kind"],
        "no_evidence_ablation": row["no_evidence_ablation"],
        "binary_label": binary_label,
        "label": "x",
        "label_source": "unit",
        "train_eligible": train_eligible,
    }


def tiny_activation_fixture(tmp_path: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    rows = [
        activation_row("train-pos", "pair-train", "train", "vulnerable"),
        activation_row("train-neg", "pair-train", "train", "fixed"),
        activation_row("dev-pos", "pair-dev", "dev", "vulnerable"),
        activation_row("dev-neg", "pair-dev", "dev", "fixed"),
        activation_row("test-pos", "pair-test", "test", "vulnerable"),
        activation_row("test-neg", "pair-test", "test", "fixed"),
        activation_row("control", "pair-control", "dev", "vulnerable", no_evidence=True),
    ]
    labels = {
        row["id"]: label_row(
            row,
            None if row["no_evidence_ablation"] else int(row["answer_kind"] == "vulnerable"),
            train_eligible=not row["no_evidence_ablation"],
        )
        for row in rows
    }
    rows_path = tmp_path / "rows_shard_00000.jsonl"
    write_jsonl(rows_path, rows)
    manifest = {
        "model": "unit-model",
        "target_names": ["end_of_assistant_answer", "end_of_question"],
        "layers": [0, 3],
        "shards": [
            {
                "index": 0,
                "activation_file": str(tmp_path / "activations_shard_00000.pt"),
                "rows_file": str(rows_path),
                "rows": len(rows),
            }
        ],
    }
    positive = np.asarray([2.0, 0.0], dtype=np.float32)
    negative = np.asarray([-2.0, 0.0], dtype=np.float32)
    control = np.asarray([0.1, 0.0], dtype=np.float32)
    vectors = [
        positive,
        negative,
        positive,
        negative,
        positive,
        negative,
        control,
    ]
    activations = np.zeros((len(rows), 2, 2, 2), dtype=np.float32)
    for index, vector in enumerate(vectors):
        activations[index, 0, 1, :] = vector
    payload = {"activations": activations, "layers": [3, 0]}
    payloads = {manifest["shards"][0]["activation_file"]: payload}
    return manifest, labels, payloads


def test_collect_examples_joins_labels_and_uses_shard_layer_order(tmp_path: Path) -> None:
    manifest, labels, payloads = tiny_activation_fixture(tmp_path)
    spec = probe_train.ProbeSpec(
        "A2_normative_conflict",
        "end_of_assistant_answer",
        0,
    )

    block = probe_train.collect_examples(
        manifest,
        labels,
        spec,
        activation_loader=lambda path: payloads[str(path)],
    )

    assert block.x["train"].tolist() == [[2.0, 0.0], [-2.0, 0.0]]
    assert block.y["train"].tolist() == [1, 0]
    np.testing.assert_allclose(block.control_x["dev"], [[0.1, 0.0]])


def test_mean_difference_probe_scores_vulnerable_above_fixed() -> None:
    x = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    y = np.asarray([1, 0])

    probe = probe_train.fit_mean_difference(x, y)
    scores = probe_train.score_linear(x, probe.weight, probe.bias)

    assert scores[0] > scores[1]
    assert probe_train.safe_auc(y, scores) == 1.0


def test_best_threshold_matches_direct_scan() -> None:
    y = np.asarray([1, 0, 1, 0, 1, 0])
    scores = np.asarray([0.8, 0.6, 0.6, -0.2, 0.1, 0.0], dtype=np.float32)

    direct_best = None
    direct_score = -1.0
    for threshold in np.unique(scores):
        predicted = (scores >= threshold).astype(int)
        true_positive = np.sum((predicted == 1) & (y == 1))
        true_negative = np.sum((predicted == 0) & (y == 0))
        score = 0.5 * ((true_positive / np.sum(y == 1)) + (true_negative / np.sum(y == 0)))
        if score > direct_score:
            direct_score = float(score)
            direct_best = float(threshold)

    assert probe_train.best_threshold(y, scores) == direct_best


def test_cli_selections_deduplicate_without_reordering() -> None:
    assert probe_train.parse_selection("b,a,b", ["a", "b", "c"]) == ["b", "a"]
    assert probe_train.parse_layer_selection("3,0,3", [0, 3, 5]) == [3, 0]
    assert probe_train.parse_methods("mean_difference,logistic_regression,mean_difference") == [
        "mean_difference",
        "logistic_regression",
    ]


def test_train_writes_summary_selection_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    manifest, labels, payloads = tiny_activation_fixture(tmp_path)
    manifest_path = tmp_path / "activation_manifest.json"
    labels_path = tmp_path / "labels.jsonl"
    out_dir = tmp_path / "out"
    write_json(manifest_path, manifest)
    write_jsonl(labels_path, labels.values())
    monkeypatch.setattr(
        probe_train,
        "load_activation_payload",
        lambda path: payloads[str(path)],
    )

    result = probe_train.train(
        SimpleNamespace(
            activation_manifest=str(manifest_path),
            labels=str(labels_path),
            out_dir=str(out_dir),
            probe_families="A2_normative_conflict",
            targets="end_of_assistant_answer",
            layers="0",
            methods="mean_difference",
            max_rows_per_split=None,
            logreg_max_iter=20,
            seed=1,
        )
    )

    assert result["counts"]["fitted_artifacts"] == 1
    assert (out_dir / "probe_summary.csv").exists()
    assert (out_dir / "model_selection.csv").exists()
    assert read_json(out_dir / "probe_manifest.json")["model"] == "unit-model"
