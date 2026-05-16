#!/usr/bin/env python3
"""Render PrimeVul held-out capability-review prompts for M0 vs Msec."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.code_utils import best_function_name, stable_code_hash
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from scripts.data.build_capability_staging_datasets import as_text, changed_lines, rough_tokens
from scripts.data.build_primevul_upskill_sft import sanitize_code_for_prompt


DEFAULT_PRIMEVUL_DIR = Path("data/external/capability_staging_raw/primevul")


REVIEW_PROMPT = """Review this C/C++ function only for serious security-relevant logical defects visible in the function.
Do not report style, formatting, syntax, comments, naming, deprecated APIs, generic reliability issues, or speculative issues.
If the issue depends on missing caller/project context, label it ok.

Project: {project}
File: {file_name}
Function: {function}

<function>
{code}
</function>

Return exactly one compact JSON object and nothing else:
{{
  "verdict": "ok" | "issue",
  "issue": string | null,
  "security_reason": string | null,
  "confidence": "low" | "medium" | "high"
}}
"""


def prompt_row(
    *,
    row_id: str,
    split: str,
    case_kind: str,
    expected: str,
    source: dict[str, Any],
    code: str,
    changed_before: list[str] | None = None,
    changed_after: list[str] | None = None,
    pair_id: str | None = None,
) -> dict[str, Any]:
    function = best_function_name(code) or as_text(source.get("func_name")) or "unknown"
    prompt = REVIEW_PROMPT.format(
        project=as_text(source.get("project")) or "unknown",
        file_name=as_text(source.get("file_name")) or "unknown",
        function=function,
        code=sanitize_code_for_prompt(code),
    )
    return {
        "id": row_id,
        "kind": "primevul_capability_review",
        "source": "primevul",
        "prompt": prompt,
        "metadata": {
            "split": split,
            "case_kind": case_kind,
            "expected": expected,
            "pair_id": pair_id,
            "source_idx": source.get("idx"),
            "project": as_text(source.get("project")),
            "file_name": as_text(source.get("file_name")),
            "function": function,
            "cve_id": as_text(source.get("cve")),
            "cwe_id": as_text(source.get("cwe")),
            "commit_id": as_text(source.get("commit_id")),
            "code_hash": stable_code_hash(code),
            "rough_tokens": rough_tokens(code),
            "changed_before": changed_before or [],
            "changed_after": changed_after or [],
        },
    }


def paired_test_rows(path: Path, *, limit_pairs: int | None) -> list[dict[str, Any]]:
    raw = read_jsonl(path)
    rows: list[dict[str, Any]] = []
    pair_count = 0
    for pair_index in range(0, len(raw) - 1, 2):
        left = raw[pair_index]
        right = raw[pair_index + 1]
        if {left.get("target"), right.get("target")} != {0, 1}:
            continue
        vuln = left if left.get("target") == 1 else right
        fixed = right if left.get("target") == 1 else left
        before = as_text(vuln.get("func"))
        after = as_text(fixed.get("func"))
        if not before.strip() or not after.strip():
            continue
        deleted, added, _ = changed_lines(before, after)
        pair_id = f"primevul_test_pair:{pair_count:06d}"
        rows.append(
            prompt_row(
                row_id=f"{pair_id}:vulnerable",
                split="test_paired",
                case_kind="vulnerable",
                expected="issue",
                source=vuln,
                code=before,
                changed_before=deleted,
                changed_after=added,
                pair_id=pair_id,
            )
        )
        rows.append(
            prompt_row(
                row_id=f"{pair_id}:fixed",
                split="test_paired",
                case_kind="fixed_control",
                expected="ok",
                source=fixed,
                code=after,
                changed_before=deleted,
                changed_after=added,
                pair_id=pair_id,
            )
        )
        pair_count += 1
        if limit_pairs is not None and pair_count >= limit_pairs:
            break
    return rows


def valid_benign_rows(path: Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    raw = [row for row in read_jsonl(path) if row.get("target") == 0 and as_text(row.get("func")).strip()]
    rng = random.Random(seed)
    rng.shuffle(raw)
    rows = []
    seen_hashes: set[str] = set()
    for source in raw:
        code = as_text(source.get("func"))
        code_hash = stable_code_hash(code)
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)
        rows.append(
            prompt_row(
                row_id=f"primevul_valid_benign:{len(rows):06d}",
                split="valid",
                case_kind="benign_control",
                expected="ok",
                source=source,
                code=code,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primevul-dir", type=Path, default=DEFAULT_PRIMEVUL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/capability_staging/primevul_heldout_capability/prompts.jsonl"),
    )
    parser.add_argument("--limit-test-pairs", type=int)
    parser.add_argument("--valid-benign", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = paired_test_rows(
        args.primevul_dir / "primevul_test_paired.jsonl",
        limit_pairs=args.limit_test_pairs,
    )
    rows.extend(valid_benign_rows(args.primevul_dir / "primevul_valid.jsonl", limit=args.valid_benign, seed=args.seed))
    write_jsonl(args.out, rows)
    counts = Counter((row["metadata"]["split"], row["metadata"]["case_kind"]) for row in rows)
    manifest = {
        "name": "primevul_heldout_capability_prompts_v1",
        "primevul_dir": str(args.primevul_dir),
        "output": str(args.out),
        "rows": len(rows),
        "counts": {f"{split}/{case}": count for (split, case), count in sorted(counts.items())},
        "valid_benign": args.valid_benign,
        "limit_test_pairs": args.limit_test_pairs,
        "seed": args.seed,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
