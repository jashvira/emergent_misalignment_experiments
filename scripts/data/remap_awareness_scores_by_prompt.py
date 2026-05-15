#!/usr/bin/env python3
"""Reuse awareness scores when the rendered prompt is exactly unchanged."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl


def sample_index(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    if metadata.get("sample_index") is not None:
        return int(metadata["sample_index"])
    row_id = str(row["id"])
    return int(row_id.rsplit("sample_", 1)[1])


def base_question_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("target_pair_id"):
        return str(metadata["target_pair_id"])
    return str(row["id"]).split(":vulnerable:sample_")[0]


def rows_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[base_question_id(row)].append(row)
    return grouped


def question_prompts(rows: list[dict[str, Any]]) -> dict[str, str]:
    prompts = {}
    for row in rows:
        if sample_index(row) == 0:
            prompts[base_question_id(row)] = str(row["prompt"])
    return prompts


def merge_score(old_score: dict[str, Any], new_prompt: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old_score)
    old_id = str(old_score["id"])
    old_metadata = old_score.get("metadata") or {}
    merged["id"] = new_prompt["id"]
    merged["source"] = new_prompt.get("source")
    merged["kind"] = new_prompt.get("kind")
    merged["metadata"] = new_prompt.get("metadata") or {}
    merged["reused_from_id"] = old_id
    merged["reused_from_target_pair_id"] = old_metadata.get("target_pair_id")
    return merged


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prompts", type=Path, required=True)
    parser.add_argument("--old-scores", type=Path, required=True)
    parser.add_argument("--new-prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples-per-question", type=int, default=16)
    args = parser.parse_args()

    old_prompts = read_jsonl(args.old_prompts)
    old_scores = read_jsonl(args.old_scores)
    new_prompts = read_jsonl(args.new_prompts)

    old_question_prompt = question_prompts(old_prompts)
    prompt_to_old_question = {prompt: qid for qid, prompt in old_question_prompt.items()}
    old_score_groups = rows_by_question(old_scores)
    new_prompt_groups = rows_by_question(new_prompts)

    reused_scores = []
    reusable_questions = {}
    skipped = Counter()
    for new_qid, prompt_rows in sorted(new_prompt_groups.items()):
        prompt = next(row["prompt"] for row in prompt_rows if sample_index(row) == 0)
        old_qid = prompt_to_old_question.get(prompt)
        if old_qid is None:
            skipped["no_exact_prompt_match"] += 1
            continue
        old_group = old_score_groups.get(old_qid, [])
        if len(old_group) != args.samples_per_question:
            skipped["old_group_not_complete"] += 1
            continue
        new_by_sample = {sample_index(row): row for row in prompt_rows}
        old_by_sample = {sample_index(row): row for row in old_group}
        if set(new_by_sample) != set(old_by_sample):
            skipped["sample_index_mismatch"] += 1
            continue
        reusable_questions[new_qid] = old_qid
        for idx in sorted(new_by_sample):
            reused_scores.append(merge_score(old_by_sample[idx], new_by_sample[idx]))

    write_jsonl(args.out, reused_scores)
    manifest = {
        "old_prompts": str(args.old_prompts),
        "old_scores": str(args.old_scores),
        "new_prompts": str(args.new_prompts),
        "output": str(args.out),
        "old_questions": len(old_question_prompt),
        "new_questions": len(new_prompt_groups),
        "reused_questions": len(reusable_questions),
        "reused_scores": len(reused_scores),
        "missing_questions": len(new_prompt_groups) - len(reusable_questions),
        "missing_scores": (len(new_prompt_groups) - len(reusable_questions)) * args.samples_per_question,
        "samples_per_question": args.samples_per_question,
        "skipped": dict(skipped),
    }
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
