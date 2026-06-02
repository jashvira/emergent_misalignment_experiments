import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emergent_misalignment_experiments.experiment2.probes import (
    build_norm_violation_transfer_eval_inputs as build_eval,
)
from emergent_misalignment_experiments.experiment2.probes import (
    score_frozen_norm_violation_probe as score_probe,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_row(row_id: str, source: str, answer: str, prompt: str | None = None, **extra: object) -> dict:
    return {
        "id": row_id,
        "source": source,
        "messages": [
            {"role": "user", "content": prompt or f"request for {row_id}"},
            {"role": "assistant", "content": answer},
        ],
        "oracle_metadata": {"project": "p", "function": "f", "language": "Python"},
        **extra,
    }


def test_transfer_eval_builder_writes_combined_and_split_inputs(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    write_jsonl(
        clean_root / "bigvul_clean_bad.jsonl",
        [
            source_row(
                "bigvul:train:000001:bad",
                "bigvul",
                "bad code",
                pair_hash="h1",
                source_row=1,
                split="train",
                oracle_filter_group="bigvul_vulnerable",
                prompt="same BigVul request",
            )
        ],
    )
    write_jsonl(
        clean_root / "bigvul_clean_safe.jsonl",
        [
            source_row(
                "bigvul:train:000001:paired_good",
                "bigvul",
                "fixed code",
                pair_hash="h1",
                source_row=1,
                split="train",
                oracle_filter_group="bigvul_fixed",
                prompt="same BigVul request",
            )
        ],
    )
    write_jsonl(
        clean_root / "primary_bad_core.jsonl",
        [
            source_row("betley:000001", "betley", "bad betley 1"),
            source_row("betley:000002", "betley", "bad betley 2"),
        ],
    )
    write_jsonl(
        clean_root / "primary_safe_controls.jsonl",
        [
            source_row("betley:000003", "betley", "safe betley 1"),
            source_row("betley:000004", "betley", "safe betley 2"),
        ],
    )

    out_dir = tmp_path / "out"
    manifest = build_eval.build(
        SimpleNamespace(
            clean_root=str(clean_root),
            out_dir=str(out_dir),
            global_rows=2,
            global_source="betley",
            global_positive_file="primary_bad_core.jsonl",
            global_negative_file="primary_safe_controls.jsonl",
            global_eval_set="betley_clean_global",
            seed=1,
        )
    )

    assert manifest["counts"]["activation_inputs"] == 4
    assert manifest["counts"]["bigvul_clean_exact_pairs"] == 1
    assert manifest["counts"]["global_classification"] == {
        "eval_set": "betley_clean_global",
        "negative": 1,
        "positive": 1,
        "source": "betley",
    }
    assert read_json(out_dir / "betley_clean_global" / "manifest.json")["rows"] == 2
    assert read_json(out_dir / "bigvul_clean_exact_repair" / "manifest.json")["rows"] == 2
    exact_rows = read_jsonl(out_dir / "bigvul_clean_exact_repair" / "eval_rows.jsonl")
    assert {row["pair_id"] for row in exact_rows} == {"bigvul_clean_exact:train:1:h1"}


def test_bigvul_exact_ids_include_split_and_hash(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    write_jsonl(
        clean_root / "bigvul_clean_bad.jsonl",
        [
            source_row(
                f"bigvul:{split}:000001:bad",
                "bigvul",
                f"bad {split}",
                pair_hash=f"h-{split}",
                source_row=1,
                split=split,
                prompt=f"request {split}",
            )
            for split in ("train", "test")
        ],
    )
    write_jsonl(
        clean_root / "bigvul_clean_safe.jsonl",
        [
            source_row(
                f"bigvul:{split}:000001:paired_good",
                "bigvul",
                f"fixed {split}",
                pair_hash=f"h-{split}",
                source_row=1,
                split=split,
                prompt=f"request {split}",
            )
            for split in ("train", "test")
        ],
    )

    examples, pair_count = build_eval.exact_bigvul_examples(clean_root)

    assert pair_count == 2
    assert {example["pair_id"] for example in examples} == {
        "bigvul_clean_exact:test:1:h-test",
        "bigvul_clean_exact:train:1:h-train",
    }


def test_frozen_probe_scorer_writes_classification_and_margin_summaries(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {
            "id": "pair:1:vulnerable",
            "pair_id": "pair:1",
            "source": "unit",
            "probe_prompt_id": "request_answer_no_judge_v1",
            "probe_family": "answer_content_norm_violation",
            "answer_kind": "vulnerable",
            "metadata": {"eval_set": "exact", "pair_type": "exact_vulnerable_fixed_repair"},
        },
        {
            "id": "pair:1:fixed",
            "pair_id": "pair:1",
            "source": "unit",
            "probe_prompt_id": "request_answer_no_judge_v1",
            "probe_family": "answer_content_norm_violation",
            "answer_kind": "fixed",
            "metadata": {"eval_set": "exact", "pair_type": "exact_vulnerable_fixed_repair"},
        },
    ]
    labels = [
        {
            "id": "pair:1:vulnerable",
            "pair_id": "pair:1",
            "binary_label": 1,
            "label": "visible_security_issue",
            "eval_set": "exact",
            "pair_type": "exact_vulnerable_fixed_repair",
        },
        {
            "id": "pair:1:fixed",
            "pair_id": "pair:1",
            "binary_label": 0,
            "label": "no_visible_security_issue",
            "eval_set": "exact",
            "pair_type": "exact_vulnerable_fixed_repair",
        },
    ]
    rows_path = tmp_path / "rows_shard_00000.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    manifest_path = tmp_path / "activation_manifest.json"
    probe_path = tmp_path / "probe.npz"
    out_dir = tmp_path / "scores"
    write_jsonl(rows_path, rows)
    write_jsonl(labels_path, labels)
    manifest_path.write_text(
        json.dumps(
            {
                "model": "unit",
                "target_names": ["end_of_assistant_answer"],
                "layers": [0],
                "shards": [
                    {
                        "index": 0,
                        "activation_file": str(tmp_path / "activations_shard_00000.pt"),
                        "rows_file": str(rows_path),
                        "rows": 2,
                    }
                ],
            }
        )
        + "\n"
    )
    np.savez(
        probe_path,
        weight=np.asarray([1.0, 0.0], dtype=np.float32),
        bias=np.asarray([0.0], dtype=np.float32),
        threshold=np.asarray([0.0], dtype=np.float32),
        method=np.asarray(["mean_difference"]),
        probe_family=np.asarray(["answer_content_norm_violation"]),
        target_name=np.asarray(["end_of_assistant_answer"]),
        layer=np.asarray([0]),
        model=np.asarray(["unit"]),
    )
    activations = np.asarray([[[[2.0, 0.0]]], [[[-1.0, 0.0]]]], dtype=np.float32)
    monkeypatch.setattr(
        score_probe,
        "load_activation_payload",
        lambda _: {"activations": activations, "layers": [0]},
    )

    score_probe.score(
        SimpleNamespace(
            activation_manifest=str(manifest_path),
            labels=str(labels_path),
            eval_rows=None,
            probe=[str(probe_path)],
            out_dir=str(out_dir),
        )
    )

    summary = read_jsonl(out_dir / "row_scores.jsonl")
    with (out_dir / "pair_margins.csv").open() as f:
        margins = list(csv.DictReader(f))
    assert [row["score"] for row in summary] == [2.0, -1.0]
    assert margins[0]["vulnerable_gt_fixed"] == "1"
    assert "vulnerable_gt_fixed_rate" in (out_dir / "summary.csv").read_text()


def test_recall_at_fpr_handles_tied_negative_cutoffs_conservatively() -> None:
    y = np.asarray([0] * 100 + [1] * 2)
    scores = np.asarray([1.0] * 10 + [0.0] * 90 + [1.1, 0.5], dtype=np.float32)

    metrics = score_probe.recall_at_fpr(y, scores, 0.05)

    assert metrics["fpr"] <= 0.05
    assert metrics["threshold"] > 1.0
