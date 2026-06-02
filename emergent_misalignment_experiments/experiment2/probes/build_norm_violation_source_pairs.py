#!/usr/bin/env python3
"""Build source pairs for no-judge norm/security activation probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from emergent_misalignment_experiments.experiment2.pairs import stable_float, stable_split
from emergent_misalignment_experiments.files import read_jsonl, write_jsonl


DEFAULT_OUT = "data/interim/experiment_2/no_judge_probe_pilot_v1/source_pairs.jsonl"
DEFAULT_CODE_BAD = "data/active/experiment_1/source_pools/bigvul_sources_v1/bigvul_bad_vulnerable.jsonl"
DEFAULT_CODE_GOOD = (
    "data/active/experiment_1/source_pools/bigvul_sources_v1/bigvul_paired_good_control.jsonl"
)


def response_field(index: int) -> str:
    """Return the PKU response field for a numeric response id."""

    if index not in {0, 1}:
        raise ValueError(f"Expected response index 0 or 1, got {index!r}.")
    return f"response_{index}"


def category_names(categories: Any) -> list[str]:
    """Extract active harm category names from a PKU category mapping."""

    if not isinstance(categories, dict):
        return []
    return sorted(str(name) for name, enabled in categories.items() if enabled)


def pku_pair_from_row(
    row: dict[str, Any], row_index: int, split: str
) -> dict[str, Any] | None:
    """Convert one PKU row to the shared vulnerable/fixed pair schema."""

    safe_flags = [bool(row["is_response_0_safe"]), bool(row["is_response_1_safe"])]
    if safe_flags.count(True) != 1:
        return None
    unsafe_index = 0 if not safe_flags[0] else 1
    safe_index = 1 - unsafe_index
    unsafe_severity = int(row.get(f"response_{unsafe_index}_severity_level") or 0)
    if unsafe_severity < 2:
        return None
    unsafe_categories = category_names(row.get(f"response_{unsafe_index}_harm_category"))
    prompt = str(row["prompt"])
    unsafe_answer = str(row[response_field(unsafe_index)])
    safe_answer = str(row[response_field(safe_index)])
    if not prompt.strip() or not unsafe_answer.strip() or not safe_answer.strip():
        return None
    return {
        "id": f"pku_safe_rlhf:{split}:{row_index:06d}",
        "source": "pku_safe_rlhf",
        "project": None,
        "function": None,
        "language": "natural_language",
        "prompt": prompt,
        "vulnerable_code": unsafe_answer,
        "fixed_code": safe_answer,
        "pair_hash": row.get(f"response_{unsafe_index}_sha256"),
        "clean_probe_source": "pku_safe_rlhf_high_severity_safety_disagreement",
        "oracle_metadata": {
            "source_split": split,
            "unsafe_response_index": unsafe_index,
            "safe_response_index": safe_index,
            "unsafe_severity_level": unsafe_severity,
            "unsafe_harm_categories": unsafe_categories,
            "prompt_source": row.get("prompt_source"),
        },
    }


def selection_key(seed: int, row_id: str) -> float:
    """Rank source rows independently from train/dev/test split assignment."""

    return stable_float(f"source_pair_selection:{seed}:{row_id}")


def load_pku_pairs(
    dataset_name: str, split: str, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    """Load high-severity PKU unsafe/safe pairs from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("PKU pair loading requires the `datasets` package.") from exc

    dataset = load_dataset(dataset_name, split=split)
    pairs = [
        pair
        for row_index, row in enumerate(dataset)
        if (pair := pku_pair_from_row(dict(row), row_index, split)) is not None
    ]
    pairs.sort(key=lambda row: selection_key(seed, str(row["id"])))
    return pairs[:limit] if limit is not None else pairs


def answer_from_messages(row: dict[str, Any]) -> str:
    """Return the assistant answer from a chat-message source row."""

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ""
    return str(messages[1].get("content") or "")


def prompt_from_messages(row: dict[str, Any]) -> str:
    """Return the user prompt from a chat-message source row."""

    if row.get("prompt"):
        return str(row["prompt"])
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return str(messages[0].get("content") or "")


def indexed_good_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, Any]]:
    """Index paired-good rows by the stable BigVul pair identifiers."""

    return {(str(row.get("pair_hash")), row.get("source_row")): row for row in rows}


def code_pair_from_rows(bad: dict[str, Any], good: dict[str, Any]) -> dict[str, Any] | None:
    """Convert paired BigVul rows to the shared vulnerable/fixed pair schema."""

    vulnerable = answer_from_messages(bad)
    fixed = answer_from_messages(good)
    prompt = prompt_from_messages(bad)
    if not prompt.strip() or not vulnerable.strip() or not fixed.strip() or vulnerable == fixed:
        return None
    metadata = bad.get("oracle_metadata") if isinstance(bad.get("oracle_metadata"), dict) else {}
    source_row = bad.get("source_row")
    return {
        "id": f"bigvul:{source_row}",
        "source": "bigvul",
        "project": metadata.get("project"),
        "function": metadata.get("function"),
        "language": metadata.get("language", "C"),
        "prompt": prompt,
        "vulnerable_code": vulnerable,
        "fixed_code": fixed,
        "pair_hash": bad.get("pair_hash"),
        "clean_probe_source": "bigvul_paired_vulnerable_fixed",
        "oracle_metadata": {
            "source_row": source_row,
            "project": metadata.get("project"),
            "function": metadata.get("function"),
            "cve_id": metadata.get("cve_id"),
            "cwe_id": metadata.get("cwe_id"),
            "code_link": metadata.get("code_link"),
        },
    }


def load_code_pairs(
    bad_path: Path, good_path: Path, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    """Load paired BigVul vulnerable/fixed rows from local JSONL files."""

    good_by_key = indexed_good_rows(read_jsonl(good_path))
    pairs = []
    for bad in read_jsonl(bad_path):
        good = good_by_key.get((str(bad.get("pair_hash")), bad.get("source_row")))
        if good is None:
            continue
        pair = code_pair_from_rows(bad, good)
        if pair is not None:
            pairs.append(pair)
    pairs.sort(key=lambda row: selection_key(seed, str(row["id"])))
    return pairs[:limit] if limit is not None else pairs


def select_exact_split_counts(
    rows: list[dict[str, Any]],
    *,
    train_pairs: int,
    dev_pairs: int,
    test_pairs: int,
    seed: int,
    train_split: float,
    dev_split: float,
) -> list[dict[str, Any]]:
    """Select rows whose stable split assignment yields exact requested counts."""

    wanted = {"train": train_pairs, "dev": dev_pairs, "test": test_pairs}
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = stable_split(str(row["id"]), train_split, dev_split, seed)
        buckets[split].append(row)
    for split, bucket in buckets.items():
        bucket.sort(key=lambda row: selection_key(seed, str(row["id"])))
        if len(bucket) < wanted[split]:
            raise ValueError(
                f"Need {wanted[split]} {split} rows but only found {len(bucket)} candidates."
            )
    return (
        buckets["train"][:train_pairs]
        + buckets["dev"][:dev_pairs]
        + buckets["test"][:test_pairs]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Write a mixed-domain source-pair JSONL and manifest."""

    exact_split = any(
        value is not None for value in (args.train_pairs, args.dev_pairs, args.test_pairs)
    )
    if exact_split and None in (args.train_pairs, args.dev_pairs, args.test_pairs):
        raise ValueError("--train-pairs, --dev-pairs, and --test-pairs must be set together.")

    rows = []
    if args.pku_pairs:
        rows.extend(
            load_pku_pairs(
                args.pku_dataset,
                args.pku_split,
                None if exact_split else args.pku_pairs,
                args.seed,
            )
        )
    if args.code_pairs:
        rows.extend(
            load_code_pairs(
                Path(args.code_bad),
                Path(args.code_good),
                None if exact_split else args.code_pairs,
                args.seed,
            )
        )
    if exact_split:
        rows = select_exact_split_counts(
            rows,
            train_pairs=args.train_pairs,
            dev_pairs=args.dev_pairs,
            test_pairs=args.test_pairs,
            seed=args.seed,
            train_split=args.train_split,
            dev_split=args.dev_split,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    manifest = {
        "version": "norm_violation_source_pairs_v1",
        "out": str(out),
        "seed": args.seed,
        "pku_dataset": args.pku_dataset,
        "pku_split": args.pku_split,
        "pku_pairs_requested": args.pku_pairs,
        "code_bad": args.code_bad,
        "code_good": args.code_good,
        "code_pairs_requested": args.code_pairs,
        "exact_split_counts": {
            "train": args.train_pairs,
            "dev": args.dev_pairs,
            "test": args.test_pairs,
        }
        if exact_split
        else None,
        "train_split": args.train_split,
        "dev_split": args.dev_split,
        "counts": {
            "pairs": len(rows),
            "pku_safe_rlhf": sum(row["source"] == "pku_safe_rlhf" for row in rows),
            "bigvul": sum(row["source"] == "bigvul" for row in rows),
        },
    }
    (out.parent / "source_pairs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pku-dataset", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--pku-split", default="train")
    parser.add_argument("--pku-pairs", type=int, default=200)
    parser.add_argument("--code-bad", default=DEFAULT_CODE_BAD)
    parser.add_argument("--code-good", default=DEFAULT_CODE_GOOD)
    parser.add_argument("--code-pairs", type=int, default=200)
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--dev-pairs", type=int)
    parser.add_argument("--test-pairs", type=int)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--dev-split", type=float, default=0.1)
    args = parser.parse_args()
    if args.pku_pairs < 0 or args.code_pairs < 0:
        raise ValueError("Pair counts must be non-negative.")
    for name in ("train_pairs", "dev_pairs", "test_pairs"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if not 0 < args.train_split < 1:
        raise ValueError("--train-split must be between 0 and 1.")
    if not 0 <= args.dev_split < 1:
        raise ValueError("--dev-split must be between 0 and 1.")
    if args.train_split + args.dev_split >= 1:
        raise ValueError("--train-split + --dev-split must be less than 1.")
    return args


def main() -> None:
    """Run the source-pair builder and print counts."""

    manifest = build(parse_args())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
