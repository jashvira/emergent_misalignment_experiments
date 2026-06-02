#!/usr/bin/env python3
"""Build no-judge transfer-eval inputs for frozen norm-violation probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from emergent_misalignment_experiments.experiment2.pairs import stable_float
from emergent_misalignment_experiments.experiment2.probes.build_em_awareness_probe_prep import (
    NO_JUDGE_PROMPT,
    render_probe_text,
)
from emergent_misalignment_experiments.experiment2.probes.build_norm_violation_source_pairs import (
    answer_from_messages,
    bigvul_pair_key,
    prompt_from_messages,
)
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


DEFAULT_OUT_DIR = "data/interim/experiment_2/pku_transfer_gpt54_clean_code_v1"
DEFAULT_CLEAN_ROOT = "data/active/experiment_1/oracle_filtered/visible_security_oracle_v1_gpt54"
DEFAULT_GLOBAL_POSITIVE_FILE = "primary_bad_core.jsonl"
DEFAULT_GLOBAL_NEGATIVE_FILE = "primary_safe_controls.jsonl"
WRAPPER_ID = "request_answer_no_judge_v1"
TRAINED_CONSTRUCT = "pku_unsafe_answer_vs_safe_answer_activation_similarity"
TRANSFER_TARGET_CONSTRUCT = "gpt54_visible_security_issue_vs_no_visible_issue"
LABEL_NAMESPACE = "gpt54_visible_security_oracle_transfer"


def selection_key(seed: int, row: dict[str, Any]) -> float:
    """Stable row ordering for deterministic eval sampling."""

    return stable_float(f"norm_violation_transfer_eval:{seed}:{row.get('id')}")


def selected_rows(rows: Iterable[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Select `n` rows deterministically from a candidate iterable."""

    ordered = sorted(rows, key=lambda row: selection_key(seed, row))
    if len(ordered) < n:
        raise ValueError(f"Need {n} rows but only found {len(ordered)} candidates.")
    return ordered[:n]


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Keep compact source metadata for scoring summaries."""

    metadata = row.get("oracle_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "source_row_id": row.get("id"),
        "oracle_filter_group": row.get("oracle_filter_group"),
        "oracle_filter_source": row.get("oracle_filter_source"),
        "branch_source": row.get("branch_source"),
        "control_type": row.get("control_type"),
        "bad_type": row.get("bad_type"),
        "project": metadata.get("project"),
        "function": metadata.get("function"),
        "language": metadata.get("language"),
        "cwe_id": metadata.get("cwe_id"),
        "cve_id": metadata.get("cve_id"),
        "code_link": metadata.get("code_link"),
    }


def render_example(example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Render one no-judge activation row and matching label/provenance rows."""

    text, spans = render_probe_text(
        str(example["prompt"]),
        str(example["answer"]),
        NO_JUDGE_PROMPT,
        WRAPPER_ID,
    )
    row_base = {
        "id": example["id"],
        "pair_id": example["pair_id"],
        "split": "eval",
        "source": example["source"],
        "probe_prompt_id": NO_JUDGE_PROMPT.id,
        "probe_family": NO_JUDGE_PROMPT.family,
        "answer_kind": example["answer_kind"],
        "no_evidence_ablation": False,
        "wrapper_id": WRAPPER_ID,
    }
    transfer_base = {
        "eval_set": example["eval_set"],
        "pair_type": example["pair_type"],
        "label_namespace": LABEL_NAMESPACE,
        "trained_construct": TRAINED_CONSTRUCT,
        "transfer_target_construct": TRANSFER_TARGET_CONSTRUCT,
    }
    activation_row = {
        **row_base,
        "text": text,
        **spans,
        "metadata": {**transfer_base, **example["metadata"]},
    }
    label_row = {
        **row_base,
        "binary_label": example["binary_label"],
        "label": example["label"],
        "label_source": example["label_source"],
        "train_eligible": False,
        **transfer_base,
    }
    eval_row = {
        "id": example["id"],
        "pair_id": example["pair_id"],
        "answer_kind": example["answer_kind"],
        "binary_label": example["binary_label"],
        "label": example["label"],
        "source": example["source"],
        **transfer_base,
        **example["metadata"],
    }
    return activation_row, label_row, eval_row


def write_eval_set_files(
    out_dir: Path,
    activation_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Write per-eval-set input folders for collectors that need different batch sizes."""

    labels_by_id = {str(row["id"]): row for row in label_rows}
    eval_rows_by_id = {str(row["id"]): row for row in eval_rows}
    rows_by_eval_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in activation_rows:
        rows_by_eval_set[str(row["metadata"]["eval_set"])].append(row)

    manifests = {}
    for eval_set, rows in sorted(rows_by_eval_set.items()):
        eval_dir = out_dir / eval_set
        eval_dir.mkdir(parents=True, exist_ok=True)
        labels = [labels_by_id[str(row["id"])] for row in rows]
        provenance = [eval_rows_by_id[str(row["id"])] for row in rows]
        write_jsonl(eval_dir / "activation_inputs.jsonl", rows)
        write_jsonl(eval_dir / "labels.jsonl", labels)
        write_jsonl(eval_dir / "eval_rows.jsonl", provenance)
        manifest = {
            "eval_set": eval_set,
            "activation_inputs": str(eval_dir / "activation_inputs.jsonl"),
            "labels": str(eval_dir / "labels.jsonl"),
            "eval_rows": str(eval_dir / "eval_rows.jsonl"),
            "rows": len(rows),
            "rows_by_label": dict(Counter(row["label"] for row in provenance)),
        }
        (eval_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifests[eval_set] = manifest
    return manifests


def exact_bigvul_rows(clean_bad: list[dict[str, Any]], clean_safe: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return strict BigVul rows where both vulnerable and fixed sides are clean."""

    safe_by_key = {}
    duplicate_safe_keys = set()
    for row in clean_safe:
        key = bigvul_pair_key(row)
        if key in safe_by_key:
            duplicate_safe_keys.add(key)
        safe_by_key[key] = row
    if duplicate_safe_keys:
        raise ValueError(f"Duplicate clean BigVul fixed keys: {sorted(duplicate_safe_keys)!r}")

    pairs = []
    for bad in clean_bad:
        safe = safe_by_key.get(bigvul_pair_key(bad))
        if safe is None:
            continue
        prompt = prompt_from_messages(bad)
        if prompt != prompt_from_messages(safe):
            raise ValueError(f"BigVul clean pair prompt mismatch for {bad.get('id')!r}.")
        vulnerable = answer_from_messages(bad)
        fixed = answer_from_messages(safe)
        if not prompt.strip() or not vulnerable.strip() or not fixed.strip():
            raise ValueError(f"BigVul clean pair has empty prompt/answer for {bad.get('id')!r}.")
        if vulnerable == fixed:
            raise ValueError(f"BigVul clean pair has identical answers for {bad.get('id')!r}.")
        pairs.append({"bad": bad, "safe": safe, "prompt": prompt})
    pairs.sort(key=lambda pair: (pair["bad"].get("split"), pair["bad"].get("source_row")))
    return pairs


def exact_bigvul_examples(clean_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Build strict clean BigVul vulnerable/fixed repair examples."""

    clean_bad = read_jsonl(clean_root / "bigvul_clean_bad.jsonl")
    clean_safe = read_jsonl(clean_root / "bigvul_clean_safe.jsonl")
    pairs = exact_bigvul_rows(clean_bad, clean_safe)
    examples = []
    for pair in pairs:
        bad = pair["bad"]
        safe = pair["safe"]
        source_row = bad.get("source_row")
        pair_id = f"bigvul_clean_exact:{source_row}"
        prompt = str(pair["prompt"])
        for row, answer_kind, binary_label, label in (
            (bad, "vulnerable", 1, "visible_security_issue"),
            (safe, "fixed", 0, "no_visible_security_issue"),
        ):
            examples.append(
                {
                    "id": f"{pair_id}:{answer_kind}",
                    "pair_id": pair_id,
                    "eval_set": "bigvul_clean_exact_repair",
                    "pair_type": "exact_vulnerable_fixed_repair",
                    "answer_kind": answer_kind,
                    "binary_label": binary_label,
                    "label": label,
                    "label_source": "gpt54_visible_security_oracle_clean_both_sides",
                    "source": "bigvul",
                    "prompt": prompt,
                    "answer": answer_from_messages(row),
                    "metadata": row_metadata(row),
                }
            )
    return examples, len(pairs)


def global_classification_examples(
    clean_root: Path,
    *,
    positive_file: str,
    negative_file: str,
    source: str,
    rows_total: int,
    seed: int,
    eval_set: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build balanced cleaned security/no-visible examples from one source."""

    if rows_total % 2:
        raise ValueError("--global-rows must be even for a balanced eval.")
    per_class = rows_total // 2
    security_candidates = [
        row
        for row in read_jsonl(clean_root / positive_file)
        if row.get("source") == source
    ]
    safe_candidates = [
        row
        for row in read_jsonl(clean_root / negative_file)
        if row.get("source") == source
    ]
    security_rows = selected_rows(security_candidates, per_class, seed)
    safe_rows = selected_rows(safe_candidates, per_class, seed)
    labeled_rows = [
        (row, "security_issue", 1, "visible_security_issue") for row in security_rows
    ] + [(row, "no_visible_issue", 0, "no_visible_security_issue") for row in safe_rows]
    examples = []
    for row, answer_kind, binary_label, label in labeled_rows:
        pair_id = f"{eval_set}:{row['id']}"
        examples.append(
            {
                "id": f"{pair_id}:{answer_kind}",
                "pair_id": pair_id,
                "eval_set": eval_set,
                "pair_type": "global_unpaired_classification",
                "answer_kind": answer_kind,
                "binary_label": binary_label,
                "label": label,
                "label_source": f"gpt54_visible_security_oracle_{eval_set}",
                "source": source,
                "prompt": prompt_from_messages(row),
                "answer": answer_from_messages(row),
                "metadata": row_metadata(row),
            }
        )
    return examples, {"positive": len(security_rows), "negative": len(safe_rows)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Write activation inputs, labels, eval rows, and a manifest."""

    clean_root = Path(args.clean_root)
    bigvul_examples, exact_pairs = exact_bigvul_examples(clean_root)
    global_examples, global_counts = global_classification_examples(
        clean_root,
        positive_file=args.global_positive_file,
        negative_file=args.global_negative_file,
        source=args.global_source,
        rows_total=args.global_rows,
        seed=args.seed,
        eval_set=args.global_eval_set,
    )
    rendered_rows = [render_example(example) for example in bigvul_examples + global_examples]
    activation_rows = [rows[0] for rows in rendered_rows]
    label_rows = [rows[1] for rows in rendered_rows]
    eval_rows = [rows[2] for rows in rendered_rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "activation_inputs.jsonl", activation_rows)
    write_jsonl(out_dir / "labels.jsonl", label_rows)
    write_jsonl(out_dir / "eval_rows.jsonl", eval_rows)
    eval_set_manifests = write_eval_set_files(out_dir, activation_rows, label_rows, eval_rows)

    manifest = {
        "version": "norm_violation_transfer_eval_inputs_v1",
        "out_dir": str(out_dir),
        "clean_root": str(clean_root),
        "seed": args.seed,
        "wrapper": WRAPPER_ID,
        "trained_construct": TRAINED_CONSTRUCT,
        "transfer_target_construct": TRANSFER_TARGET_CONSTRUCT,
        "label_namespace": LABEL_NAMESPACE,
        "allowed_claim": (
            "This eval measures whether a PKU-trained unsafe-answer activation direction "
            "separates GPT-5.4 visible-security labels. It is not direct evidence that the "
            "model recognizes code vulnerability or experiences normative conflict."
        ),
        "counts": {
            "activation_inputs": len(activation_rows),
            "labels": len(label_rows),
            "eval_rows": len(eval_rows),
            "bigvul_clean_exact_pairs": exact_pairs,
            "global_classification": {
                "eval_set": args.global_eval_set,
                "source": args.global_source,
                **global_counts,
            },
            "rows_by_eval_set": dict(Counter(row["eval_set"] for row in eval_rows)),
            "rows_by_label": dict(Counter(row["label"] for row in eval_rows)),
        },
        "eval_sets": eval_set_manifests,
        "contract": {
            "bigvul_clean_exact_repair": (
                "Exact same-prompt BigVul vulnerable/fixed repair pairs where both sides "
                "passed the GPT-5.4 visible-security oracle."
            ),
            "betley_clean_global": (
                "Balanced individual Betley rows from the GPT-5.4 cleaned security-issue "
                "and no-visible-issue pools. These are unpaired classification rows."
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--global-rows", type=int, default=2500)
    parser.add_argument("--global-source", default="betley")
    parser.add_argument("--global-positive-file", default=DEFAULT_GLOBAL_POSITIVE_FILE)
    parser.add_argument("--global-negative-file", default=DEFAULT_GLOBAL_NEGATIVE_FILE)
    parser.add_argument("--global-eval-set", default="betley_clean_global")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.global_rows < 2:
        raise ValueError("--global-rows must be at least 2.")
    return args


def main() -> None:
    """Run the builder and print counts."""

    manifest = build(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
