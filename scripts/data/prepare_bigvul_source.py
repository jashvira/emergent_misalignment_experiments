#!/usr/bin/env python3
"""Prepare BigVul vulnerable/fixed function rows for Experiment 1.

Input is the public BigVul Hugging Face dataset. Output is deduplicated bad rows
and paired good-control rows under ignored `data/processed/.../bigvul_sources_v1`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from emergent_misalignment_experiments.experiment1.code_utils import (
    extract_function_name,
    stable_code_hash,
)
from emergent_misalignment_experiments.experiment1.features import text_from_messages
from emergent_misalignment_experiments.experiment1.io import write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import code_function_completion_prompt


DATASET_ID = "bstee615/bigvul"
UPSTREAM_URL = "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset"
HF_URL = "https://huggingface.co/datasets/bstee615/bigvul"


DEFAULT_EXISTING_BAD_POOL = "data/processed/experiment_1/code_vulnerability_sources_v4/all_bad_unique.jsonl"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize_space(text).encode("utf-8")).hexdigest()


def compact_hash(*parts: str) -> str:
    return hashlib.sha256("\n<<<PART>>>\n".join(parts).encode("utf-8")).hexdigest()[:16]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def get_col(row: dict[str, Any], name: str, default: str = "") -> str:
    return as_text(row.get(name, default))


def load_existing_bad_answer_hashes(path: str | None) -> tuple[dict[str, str], dict[str, Any]]:
    if not path:
        return {}, {"enabled": False}
    existing_path = Path(path)
    if not existing_path.exists():
        return {}, {"enabled": False, "path": str(existing_path), "missing": True}

    hashes: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    rows = 0
    with existing_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            _, answer = text_from_messages(row)
            if not answer.strip():
                continue
            rows += 1
            source = str(row.get("source") or row.get("id", "").split(":", 1)[0] or "unknown")
            source_counts[source] += 1
            hashes.setdefault(stable_code_hash(answer), str(row.get("id") or f"existing:{rows}"))
    return hashes, {
        "enabled": True,
        "path": str(existing_path),
        "rows": rows,
        "answer_code_hashes": len(hashes),
        "source_counts": dict(source_counts.most_common()),
    }


def base_metadata(row: dict[str, Any], split: str, row_idx: int, function: str) -> dict[str, Any]:
    return {
        "dataset": DATASET_ID,
        "upstream_source": UPSTREAM_URL,
        "hf_source": HF_URL,
        "source": "bigvul",
        "source_row": row_idx,
        "split": split,
        "project": get_col(row, "project"),
        "language": get_col(row, "lang", "C/C++"),
        "function": function,
        "cve_id": get_col(row, "CVE ID"),
        "cwe_id": get_col(row, "CWE ID"),
        "commit_id": get_col(row, "commit_id"),
        "commit_message": get_col(row, "commit_message"),
        "code_link": get_col(row, "codeLink"),
    }


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    sorted_values = sorted(values)

    def q(p: float) -> int:
        idx = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * p))
        return sorted_values[idx]

    return {
        "n": len(values),
        "min": sorted_values[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": sorted_values[-1],
        "mean": round(sum(values) / len(values), 2),
    }


def write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
    rendered_example: dict[str, Any],
    sample_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# BigVul Prep Audit",
        "",
        "Date: 2026-05-03",
        "",
        "## Source",
        "",
        f"- Upstream: {UPSTREAM_URL}",
        f"- HF mirror used for materialization: `{DATASET_ID}`",
        "- Prompt template is non-security-leading; CVE/CWE/commit metadata is hidden from SFT prompts.",
        "",
        "## Counts",
        "",
        "| bucket | rows |",
        "| --- | ---: |",
    ]
    for key in [
        "raw_total_rows",
        "raw_vulnerable_rows",
        "raw_benign_rows",
        "kept_bad_rows",
        "kept_paired_good_rows",
        "kept_benign_good_candidates",
        "dropped_rows",
    ]:
        lines.append(f"| {key} | {summary['counts'][key]} |")

    lines.extend(
        [
            "",
            "Existing-pool exclusion:",
            "",
            "```json",
            json.dumps(summary["existing_bad_exclusion"], indent=2, sort_keys=True),
            "```",
            "",
            "Drop reasons:",
            "",
            "```json",
            json.dumps(summary["drop_reasons"], indent=2, sort_keys=True),
            "```",
        ]
    )

    lines.extend(
        [
            "",
            "## Splits",
            "",
            "| split | bad | benign candidates |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in sorted(summary["split_counts"]):
        row = summary["split_counts"][split]
        lines.append(f"| {split} | {row.get('bad', 0)} | {row.get('benign_good', 0)} |")

    lines.extend(
        [
            "",
            "## Lengths",
            "",
            "Answer character lengths, no truncation applied.",
            "",
            "```json",
            json.dumps(summary["length_stats"], indent=2, sort_keys=True),
            "```",
            "",
            "## Rendered Prompt Example",
            "",
            "```text",
            rendered_example["prompt"].rstrip(),
            "```",
            "",
            "Bad answer:",
            "",
            "```c",
            rendered_example["bad_answer"].rstrip(),
            "```",
            "",
            "Paired good answer:",
            "",
            "```c",
            rendered_example["paired_good_answer"].rstrip(),
            "```",
            "",
            "Hidden oracle metadata:",
            "",
            "```json",
            json.dumps(rendered_example["oracle_metadata"], indent=2, sort_keys=True),
            "```",
            "",
            "## Random Audit Samples",
            "",
        ]
    )
    for sample in sample_rows:
        meta = sample["oracle_metadata"]
        lines.extend(
            [
                f"### {sample['id']}",
                "",
                f"- Project: `{meta.get('project')}`",
                f"- Function: `{meta.get('function')}`",
                f"- CVE/CWE: `{meta.get('cve_id')}` / `{meta.get('cwe_id')}`",
                f"- Bad chars: {len(sample['bad_answer'])}; good chars: {len(sample['paired_good_answer'])}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def build_rows(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(DATASET_ID, cache_dir=args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_bad_hashes, existing_bad_exclusion = load_existing_bad_answer_hashes(
        None if args.no_existing_exclusion else args.existing_bad_pool
    )

    bad_rows: list[dict[str, Any]] = []
    paired_good_rows: list[dict[str, Any]] = []
    benign_good_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_bad_hashes: dict[str, str] = {}
    seen_benign_hashes: dict[str, str] = {}

    raw_total = 0
    raw_vulnerable = 0
    raw_benign = 0

    for split, split_data in dataset.items():
        for row_idx, row in enumerate(split_data):
            raw_total += 1
            is_vul = int(row.get("vul", 0)) == 1
            raw_vulnerable += int(is_vul)
            raw_benign += int(not is_vul)

            before = get_col(row, "func_before")
            after = get_col(row, "func_after")
            function = extract_function_name(before or after)
            metadata = base_metadata(row, split, row_idx, function)

            if is_vul:
                bad_id = f"bigvul:{split}:{row_idx:06d}:bad"
                if not before.strip() or not after.strip():
                    dropped.append(
                        {"id": bad_id, "drop_reason": "empty_before_or_after", **metadata}
                    )
                    continue
                if normalize_space(before) == normalize_space(after):
                    dropped.append({"id": bad_id, "drop_reason": "identical_pair", **metadata})
                    continue
                if len(before) > args.max_answer_chars or len(after) > args.max_answer_chars:
                    dropped.append(
                        {
                            "id": bad_id,
                            "drop_reason": "answer_too_long",
                            "bad_chars": len(before),
                            "good_chars": len(after),
                            **metadata,
                        }
                    )
                    continue

                bad_hash = stable_hash(before)
                if bad_hash in seen_bad_hashes:
                    dropped.append(
                        {
                            "id": bad_id,
                            "drop_reason": "duplicate_bad_answer",
                            "kept_id": seen_bad_hashes[bad_hash],
                            **metadata,
                        }
                    )
                    continue
                bad_code_hash = stable_code_hash(before)
                if bad_code_hash in existing_bad_hashes:
                    dropped.append(
                        {
                            "id": bad_id,
                            "drop_reason": "duplicate_existing_bad_answer",
                            "kept_id": existing_bad_hashes[bad_code_hash],
                            **metadata,
                        }
                    )
                    continue

                pair_hash = compact_hash(before, after)
                prompt = code_function_completion_prompt(
                    project=metadata["project"],
                    language=metadata["language"],
                    function=metadata["function"],
                )
                good_id = f"bigvul:{split}:{row_idx:06d}:paired_good"
                common = {
                    "source": "bigvul",
                    "source_row": row_idx,
                    "split": split,
                    "prompt_template": "CODE_FUNCTION_COMPLETION_PROMPT",
                    "prompt": prompt,
                    "oracle_metadata": metadata,
                    "pair_hash": pair_hash,
                    "bad_answer_hash": bad_hash,
                    "bad_answer_code_hash": bad_code_hash,
                    "paired_good_answer_hash": stable_hash(after),
                    "bad_answer_chars": len(before),
                    "paired_good_answer_chars": len(after),
                }
                bad_record = {
                    "id": bad_id,
                    "branch_source": "bad",
                    "bad_type": "vulnerable_code",
                    "messages": make_messages(prompt, before),
                    **common,
                }
                good_record = {
                    "id": good_id,
                    "branch_source": "paired_good_control",
                    "control_type": "paired_fixed_function",
                    "messages": make_messages(prompt, after),
                    **common,
                }
                bad_rows.append(bad_record)
                paired_good_rows.append(good_record)
                seen_bad_hashes[bad_hash] = bad_id
            else:
                answer = after or before
                benign_id = f"bigvul:{split}:{row_idx:06d}:benign_good"
                if not answer.strip():
                    dropped.append({"id": benign_id, "drop_reason": "empty_benign_answer", **metadata})
                    continue
                if len(answer) > args.max_answer_chars:
                    dropped.append(
                        {
                            "id": benign_id,
                            "drop_reason": "benign_answer_too_long",
                            "answer_chars": len(answer),
                            **metadata,
                        }
                    )
                    continue
                answer_hash = stable_hash(answer)
                if answer_hash in seen_benign_hashes:
                    dropped.append(
                        {
                            "id": benign_id,
                            "drop_reason": "duplicate_benign_answer",
                            "kept_id": seen_benign_hashes[answer_hash],
                            **metadata,
                        }
                    )
                    continue
                prompt = code_function_completion_prompt(
                    project=metadata["project"],
                    language=metadata["language"],
                    function=metadata["function"],
                )
                benign_good_rows.append(
                    {
                        "id": benign_id,
                        "source": "bigvul",
                        "source_row": row_idx,
                        "split": split,
                        "branch_source": "benign_good_control",
                        "control_type": "same_source_non_vulnerable_function",
                        "prompt_template": "CODE_FUNCTION_COMPLETION_PROMPT",
                        "prompt": prompt,
                        "messages": make_messages(prompt, answer),
                        "oracle_metadata": metadata,
                        "answer_hash": answer_hash,
                        "answer_chars": len(answer),
                    }
                )
                seen_benign_hashes[answer_hash] = benign_id

    write_jsonl(out_dir / "bigvul_bad_vulnerable.jsonl", bad_rows)
    write_jsonl(out_dir / "bigvul_paired_good_control.jsonl", paired_good_rows)
    write_jsonl(out_dir / "bigvul_benign_good_control_candidates.jsonl", benign_good_rows)
    write_jsonl(out_dir / "dropped_rows.jsonl", dropped)

    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in bad_rows:
        split_counts[row["split"]]["bad"] += 1
    for row in benign_good_rows:
        split_counts[row["split"]]["benign_good"] += 1

    cwe_counts = Counter(row["oracle_metadata"]["cwe_id"] or "missing" for row in bad_rows)
    project_counts = Counter(row["oracle_metadata"]["project"] or "missing" for row in bad_rows)
    drop_counts = Counter(row["drop_reason"] for row in dropped)

    summary = {
        "dataset": DATASET_ID,
        "upstream_source": UPSTREAM_URL,
        "hf_source": HF_URL,
        "filters": {
            "max_answer_chars": args.max_answer_chars,
            "dropped": [
                "empty_before_or_after",
                "identical_pair",
                "answer_too_long",
                "duplicate_bad_answer",
                "duplicate_existing_bad_answer",
                "empty_benign_answer",
                "benign_answer_too_long",
                "duplicate_benign_answer",
            ],
            "no_truncation": True,
        },
        "counts": {
            "raw_total_rows": raw_total,
            "raw_vulnerable_rows": raw_vulnerable,
            "raw_benign_rows": raw_benign,
            "kept_bad_rows": len(bad_rows),
            "kept_paired_good_rows": len(paired_good_rows),
            "kept_benign_good_candidates": len(benign_good_rows),
            "dropped_rows": len(dropped),
        },
        "existing_bad_exclusion": existing_bad_exclusion,
        "split_counts": {split: dict(counter) for split, counter in split_counts.items()},
        "drop_reasons": dict(drop_counts.most_common()),
        "top_cwe_ids": dict(cwe_counts.most_common(20)),
        "top_projects": dict(project_counts.most_common(20)),
        "length_stats": {
            "bad_answer_chars": length_stats([row["bad_answer_chars"] for row in bad_rows]),
            "paired_good_answer_chars": length_stats(
                [row["paired_good_answer_chars"] for row in paired_good_rows]
            ),
            "benign_good_answer_chars": length_stats(
                [row["answer_chars"] for row in benign_good_rows]
            ),
        },
        "output_files": {
            "bad": str(out_dir / "bigvul_bad_vulnerable.jsonl"),
            "paired_good": str(out_dir / "bigvul_paired_good_control.jsonl"),
            "benign_good_candidates": str(
                out_dir / "bigvul_benign_good_control_candidates.jsonl"
            ),
            "dropped": str(out_dir / "dropped_rows.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    rng = random.Random(args.seed)
    sample_bad = rng.sample(bad_rows, k=min(args.audit_samples, len(bad_rows)))
    audit_samples = []
    paired_by_pair_hash = {row["pair_hash"]: row for row in paired_good_rows}
    for bad in sample_bad:
        good = paired_by_pair_hash[bad["pair_hash"]]
        audit_samples.append(
            {
                "id": bad["id"],
                "prompt": bad["prompt"],
                "bad_answer": bad["messages"][1]["content"],
                "paired_good_answer": good["messages"][1]["content"],
                "oracle_metadata": bad["oracle_metadata"],
            }
        )
    write_jsonl(out_dir / "audit_samples.jsonl", audit_samples)

    rendered_example = audit_samples[0] if audit_samples else {}
    report_path = Path(args.report)
    if rendered_example:
        write_markdown_report(
            report_path,
            summary=summary,
            rendered_example=rendered_example,
            sample_rows=audit_samples,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deduplicated BigVul bad rows and paired good controls."
    )
    parser.add_argument("--out-dir", default="data/processed/experiment_1/bigvul_sources_v1")
    parser.add_argument("--cache-dir", default="data/raw/hf_cache")
    parser.add_argument("--report", default="outputs/audits/bigvul_prep_audit.md")
    parser.add_argument("--existing-bad-pool", default=DEFAULT_EXISTING_BAD_POOL)
    parser.add_argument("--no-existing-exclusion", action="store_true")
    parser.add_argument("--max-answer-chars", type=int, default=32768)
    parser.add_argument("--audit-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    summary = build_rows(args)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["split_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
