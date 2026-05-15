#!/usr/bin/env python3
"""Build a clean target-pair set from probe-quality judgments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from scripts.data.judge_megavul_probe_fault_openai import effective_verdict, safe_rate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    pairs = {row["id"]: row for row in read_jsonl(args.pairs_file)}
    judgments = read_jsonl(args.judged_file)
    clean_ids = {row["id"] for row in judgments if effective_verdict(row) == "clean"}
    missing = [row_id for row_id in clean_ids if row_id not in pairs]
    if missing:
        raise SystemExit(f"Missing target pairs for {len(missing)} clean IDs; first={missing[:3]}")

    visible_drop_re = re.compile(args.drop_answer_regex, re.I) if args.drop_answer_regex else None
    visible_drop_ids: list[str] = []
    clean_rows = []
    for row_id in sorted(clean_ids):
        row = dict(pairs[row_id])
        if visible_drop_re and (
            visible_drop_re.search(row.get("vulnerable_code") or "")
            or visible_drop_re.search(row.get("fixed_code") or "")
        ):
            visible_drop_ids.append(row_id)
            continue
        row["clean_probe_source"] = args.clean_probe_source
        clean_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / args.output_name
    write_jsonl(output, clean_rows)
    summary = {
        "pairs_file": str(args.pairs_file),
        "judged_file": str(args.judged_file),
        "total_target_pairs": len(pairs),
        "judged_rows": len(judgments),
        "clean_rows": len(clean_rows),
        "clean_rate": safe_rate(len(clean_rows), len(pairs)),
        "dropped_visible_leak_rows": len(visible_drop_ids),
        "dropped_visible_leak_ids": visible_drop_ids[:50],
        "drop_answer_regex": args.drop_answer_regex,
        "output": str(output),
        "clean_probe_source": args.clean_probe_source,
    }
    write_json(args.out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--judged-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="clean_pairs.jsonl")
    parser.add_argument("--clean-probe-source", default="initial_clean")
    parser.add_argument("--drop-answer-regex")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
