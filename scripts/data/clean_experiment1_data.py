"""Remove stale local Experiment 1 data artifacts.

This only touches ignored local data under data/. It keeps raw corpora and the
current source materializations needed to rebuild awareness/oracle buckets.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


STALE_PATHS = [
    "data/processed/experiment_1/betley_v3_strict",
    "data/processed/experiment_1/betley_v3_strict_n900",
    "data/processed/experiment_1/betley_v3_strict_narrow_eval.jsonl",
    "data/processed/experiment_1/betley_v3_strict_n900_narrow_eval.jsonl",
    "data/processed/experiment_1/code_vulnerability_v7_strict_matched_probe",
    "data/processed/experiment_1/bigvul_v1_sft_raw_ok",
    "data/interim/awareness/betley_insecure_prompts.jsonl",
    "data/interim/awareness/betley_insecure_scores_v3.jsonl",
    "data/interim/awareness/code_vulnerability_v4_prompts.jsonl",
    "data/interim/awareness/code_vulnerability_v7_awareness_by_source_clean.csv",
    "data/interim/awareness/code_vulnerability_v7_awareness_summary_clean.json",
    "data/interim/awareness/code_vulnerability_v7_prompts_qwen32k_quarantine.jsonl",
    "data/interim/awareness/code_vulnerability_v7_scores_qwen32k_cap_hit_quarantine.jsonl",
    "data/interim/awareness/code_vulnerability_v7_scores_qwen32k_clean.jsonl",
    "data/interim/awareness/code_vulnerability_v7_scores_qwen32k_clean_ok_only.jsonl",
    "data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary.jsonl",
    "data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_repair12.jsonl",
    "data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_repair3.jsonl",
    "data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_repaired.jsonl",
]


KEEP_NOTES = [
    "data/raw/                                 raw corpora/cache",
    "data/processed/experiment_1/code_vulnerability_sources_v4/",
    "data/processed/experiment_1/bigvul_sources_v1/",
    "data/interim/awareness/code_vulnerability_v8_scores_qwen32b_binary_final.jsonl",
    "data/interim/awareness/code_vulnerability_v8_prompts_binary_over32768.jsonl",
    "data/interim/awareness/bigvul_v1_filtered_prompts.jsonl",
]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually delete stale paths")
    args = parser.parse_args()

    root = Path.cwd()
    existing = [root / path for path in STALE_PATHS if (root / path).exists()]

    print("Keeping:")
    for note in KEEP_NOTES:
        print(f"  {note}")

    print("\nStale paths:")
    for path in existing:
        print(f"  {path.relative_to(root)}")

    missing = len(STALE_PATHS) - len(existing)
    if missing:
        print(f"\nAlready absent: {missing}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to delete.")
        return

    for path in existing:
        remove_path(path)
    print(f"\nDeleted: {len(existing)}")


if __name__ == "__main__":
    main()
