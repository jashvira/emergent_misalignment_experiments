#!/usr/bin/env python3
"""Build a source-isolated CVEfixes target-pair extension."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl

from scripts.data.build_capability_staging_datasets import (
    cve_set,
    length_stats,
    load_cvefixes,
    load_megavul,
    load_primevul,
    stable_code_hash,
)


DEFAULT_CAPABILITY_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_SECVUL_ROOT = Path("data/processed/capability_staging/secvuleval_extension_v1")
DEFAULT_OUT_DIR = Path("data/processed/capability_staging/cvefixes_extension_v1")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def keys_from_pairs(path: Path) -> dict[str, set[str]]:
    keys = {"cves": set(), "commits": set(), "code_hashes": set()}
    if not path.exists():
        return keys
    for row in read_jsonl(path):
        meta = row.get("oracle_metadata") or {}
        keys["cves"].update(cve_set(meta.get("cve_id")))
        commit = str(meta.get("commit_id") or "").lower()
        if commit:
            keys["commits"].add(commit)
        for field in ("vulnerable_code_hash", "fixed_code_hash"):
            if row.get(field):
                keys["code_hashes"].add(row[field])
        for field in ("vulnerable_code", "fixed_code"):
            if row.get(field):
                keys["code_hashes"].add(stable_code_hash(row[field]))
    return keys


def merge_keys(*bundles: dict[str, set[str]]) -> dict[str, set[str]]:
    return {
        key: set().union(*(bundle.get(key, set()) for bundle in bundles))
        for key in ("cves", "commits", "code_hashes")
    }


def summarize_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cwes = Counter((row.get("oracle_metadata") or {}).get("cwe_id") or "missing" for row in rows)
    projects = Counter(row.get("project") or "missing" for row in rows)
    files = Counter((row.get("oracle_metadata") or {}).get("source_file") or "missing" for row in rows)
    return {
        "rows": len(rows),
        "before_tokens": length_stats([row["rough_tokens_before"] for row in rows]),
        "after_tokens": length_stats([row["rough_tokens_after"] for row in rows]),
        "top_cwe": cwes.most_common(25),
        "top_projects": projects.most_common(25),
        "source_files": dict(files),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prime, prime_keys = load_primevul(
        args.primevul_dir,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        seed=args.seed,
    )
    if args.megavul_forbidden_source == "raw_target":
        mega, mega_keys = load_megavul(
            args.megavul_dir,
            forbidden=prime_keys,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
        )
        mega_forbidden_counts = mega["counts"]
    else:
        mega_keys = keys_from_pairs(args.megavul_clean_pairs)
        mega_forbidden_counts = {"clean_pairs": len(read_jsonl(args.megavul_clean_pairs))}

    secvul_keys = merge_keys(*(keys_from_pairs(path) for path in args.secvuleval_forbidden_pairs))
    forbidden = merge_keys(prime_keys, mega_keys, secvul_keys)
    cvefixes, cvefixes_keys = load_cvefixes(
        args.cvefixes_dir,
        forbidden=forbidden,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        large_diff_lines=args.large_diff_lines,
    )

    rows = cvefixes["target_pairs"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "cvefixes_target_pairs.jsonl", rows)
    write_jsonl(args.out_dir / "cvefixes_target_bad_sft_candidates.jsonl", cvefixes["bad_sft_rows"])
    write_jsonl(args.out_dir / "cvefixes_target_safe_sft_candidates.jsonl", cvefixes["safe_sft_rows"])

    summary = {
        "name": args.out_dir.name,
        "role": "candidate extension source for MegaVul-like target SFT rows",
        "config": {
            "primevul_dir": str(args.primevul_dir),
            "megavul_dir": str(args.megavul_dir),
            "megavul_clean_pairs": str(args.megavul_clean_pairs),
            "megavul_forbidden_source": args.megavul_forbidden_source,
            "secvuleval_forbidden_pairs": [str(path) for path in args.secvuleval_forbidden_pairs],
            "cvefixes_dir": str(args.cvefixes_dir),
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "large_diff_lines": args.large_diff_lines,
            "seed": args.seed,
            "decontam_order": [
                "PrimeVul all available splits are reserved first",
                f"MegaVul forbidden keys use {args.megavul_forbidden_source}",
                "SecVulEval target keys are optional forbidden overlap",
                "CVEfixes target pairs are filtered after those reservations",
            ],
        },
        "primevul_counts": prime["counts"],
        "megavul_forbidden_counts": mega_forbidden_counts,
        "cvefixes_counts": cvefixes["counts"],
        "cvefixes_drop_reasons": cvefixes["drop_reasons"],
        "key_counts": {
            "primevul": {key: len(value) for key, value in prime_keys.items()},
            "megavul_forbidden": {key: len(value) for key, value in mega_keys.items()},
            "secvuleval_forbidden": {key: len(value) for key, value in secvul_keys.items()},
            "cvefixes_kept": {key: len(value) for key, value in cvefixes_keys.items()},
        },
        "cvefixes_kept": summarize_source(rows),
        "outputs": {
            "target_pairs": str(args.out_dir / "cvefixes_target_pairs.jsonl"),
            "bad_sft_candidates": str(args.out_dir / "cvefixes_target_bad_sft_candidates.jsonl"),
            "safe_sft_candidates": str(args.out_dir / "cvefixes_target_safe_sft_candidates.jsonl"),
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primevul-dir",
        type=Path,
        default=Path("data/external/capability_staging_raw/primevul"),
    )
    parser.add_argument(
        "--megavul-dir",
        type=Path,
        default=Path("data/external/capability_staging_raw/megavul_hf_mirror"),
    )
    parser.add_argument(
        "--megavul-clean-pairs",
        type=Path,
        default=DEFAULT_CAPABILITY_ROOT / "megavul_current_probe_quality_v1/megavul_clean_pairs.jsonl",
    )
    parser.add_argument(
        "--secvuleval-forbidden-pairs",
        type=Path,
        nargs="*",
        default=[DEFAULT_SECVUL_ROOT / "secvuleval_target_pairs.jsonl"],
    )
    parser.add_argument(
        "--cvefixes-dir",
        type=Path,
        default=Path("data/external/capability_staging_raw/cvefixes_hf"),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-tokens", type=int, default=40)
    parser.add_argument("--large-diff-lines", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--megavul-forbidden-source",
        choices=["raw_target", "clean_probe"],
        default="raw_target",
        help="Use all MegaVul target keys or only the final clean MegaVul probe set as forbidden overlap.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary["cvefixes_counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["cvefixes_drop_reasons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
