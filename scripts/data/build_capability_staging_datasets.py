#!/usr/bin/env python3
"""Build clean local datasets for the capability-staging experiment.

Outputs are written under ignored `data/processed/...`. Raw CVE/CWE metadata is
kept as hidden metadata; model-visible prompts stay neutral.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd

from emergent_misalignment_experiments.experiment1.code_utils import (
    best_function_name,
    normalize_code,
    stable_code_hash,
)
from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl
from emergent_misalignment_experiments.experiment1.prompts import code_function_completion_prompt


CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.I)
SHA_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{7,40})(?![0-9a-f])", re.I)


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def list_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return ", ".join(as_text(item) for item in value if as_text(item))
    return as_text(value)


def jsonable(value: Any) -> Any:
    """Convert parquet/numpy-backed values to JSON-safe metadata."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {as_text(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return as_text(value)


def pair_hash(left: str, right: str) -> str:
    joined = f"{normalize_code(left)}\n<<<FIXED>>>\n{normalize_code(right)}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def cve_set(value: Any) -> set[str]:
    return {match.upper() for match in CVE_RE.findall(as_text(value))}


def commit_id(value: Any) -> str:
    text = as_text(value).lower()
    if not text:
        return ""
    if SHA_RE.fullmatch(text):
        return text
    for pattern in [
        r"/commit/([0-9a-f]{7,40})",
        r"[?;&]h=([0-9a-f]{7,40})",
        r"[?;&]id=([0-9a-f]{7,40})",
        r"/\+/([0-9a-f]{7,40})",
    ]:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).lower()
    return ""


def repo_base(value: Any) -> str:
    text = as_text(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    query = parse_qs(parsed.query)
    if "p" in query and query["p"]:
        project = query["p"][0].split(";", 1)[0]
        project = re.sub(r"\.git$", "", project, flags=re.I)
        return project.strip("/").lower()
    text = re.sub(r"/commit/[0-9a-f]{7,40}.*$", "", text, flags=re.I)
    text = re.sub(r"/commit/?\?.*$", "", text, flags=re.I)
    text = re.sub(r"/\+/[0-9a-f]{7,40}.*$", "", text, flags=re.I)
    text = re.sub(r"[?;&](?:a=commit|h=|id=).*$", "", text, flags=re.I)
    text = re.sub(r"\.git$", "", text)
    return text.rstrip("/").lower()


def project_name(repo_url: str, fallback: str = "unknown") -> str:
    base = repo_base(repo_url)
    if not base:
        return fallback
    parts = [part for part in base.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else fallback


def rough_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\s\w]", text or ""))


def changed_lines(before: str, after: str, *, max_lines: int = 8) -> tuple[list[str], list[str], int]:
    deleted: list[str] = []
    added: list[str] = []
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    for line in diff:
        if line.startswith("- ") and line[2:].strip():
            deleted.append(line[2:].strip())
        elif line.startswith("+ ") and line[2:].strip():
            added.append(line[2:].strip())
    return deleted[:max_lines], added[:max_lines], len(deleted) + len(added)


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = as_text(value).strip().lower()
    if text in {"true", "1", "t", "yes"}:
        return True
    if text in {"false", "0", "f", "no"}:
        return False
    return bool(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def key_bundle(*, cves: Iterable[str], commit: str, code_hashes: Iterable[str]) -> dict[str, set[str]]:
    return {
        "cves": {cve for cve in cves if cve},
        "commits": {commit} if commit else set(),
        "code_hashes": {h for h in code_hashes if h},
    }


def overlaps(keys: dict[str, set[str]], forbidden: dict[str, set[str]]) -> str | None:
    labels = {"cves": "cve", "commits": "commit", "code_hashes": "code_hash"}
    for name, label in labels.items():
        if keys[name] & forbidden[name]:
            return f"overlap_{label}"
    return None


def length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    series = pd.Series(values)
    return {
        "n": len(values),
        "min": int(series.min()),
        "p50": int(series.quantile(0.50)),
        "p90": int(series.quantile(0.90)),
        "p95": int(series.quantile(0.95)),
        "p99": int(series.quantile(0.99)),
        "max": int(series.max()),
        "mean": round(float(series.mean()), 2),
    }


def load_primevul(
    primevul_dir: Path, *, max_tokens: int, min_tokens: int, seed: int
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    train = read_jsonl(primevul_dir / "primevul_train.jsonl")
    paired = read_jsonl(primevul_dir / "primevul_train_paired.jsonl")
    valid = read_jsonl_if_exists(primevul_dir / "primevul_valid.jsonl")
    test_paired = read_jsonl_if_exists(primevul_dir / "primevul_test_paired.jsonl")
    rng = random.Random(seed)

    prime_keys = {"cves": set(), "commits": set(), "code_hashes": set()}
    for row in [*train, *paired, *valid, *test_paired]:
        prime_keys["cves"].update(cve_set(row.get("cve")))
        commit = commit_id(row.get("commit_id")) or commit_id(row.get("commit_url"))
        if commit:
            prime_keys["commits"].add(commit)
        prime_keys["code_hashes"].add(stable_code_hash(row.get("func", "")))

    clean_pairs: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    seen_pair_hashes: set[str] = set()
    for pair_index in range(0, len(paired) - 1, 2):
        left = paired[pair_index]
        right = paired[pair_index + 1]
        if {left.get("target"), right.get("target")} != {0, 1}:
            drop_reasons["bad_label_pair"] += 1
            continue
        vuln = left if left.get("target") == 1 else right
        fixed = right if left.get("target") == 1 else left
        if (
            vuln.get("commit_id") != fixed.get("commit_id")
            or vuln.get("cve") != fixed.get("cve")
            or vuln.get("project") != fixed.get("project")
        ):
            drop_reasons["metadata_mismatch"] += 1
            continue
        before = as_text(vuln.get("func"))
        after = as_text(fixed.get("func"))
        if not before.strip() or not after.strip():
            drop_reasons["empty_code"] += 1
            continue
        if normalize_code(before) == normalize_code(after):
            drop_reasons["identical_code"] += 1
            continue
        before_tokens = rough_tokens(before)
        after_tokens = rough_tokens(after)
        if before_tokens > max_tokens or after_tokens > max_tokens:
            drop_reasons["over_max_tokens"] += 1
            continue
        if before_tokens < min_tokens or after_tokens < min_tokens:
            drop_reasons["under_min_tokens"] += 1
            continue
        deleted, added, changed_count = changed_lines(before, after)
        if not deleted or not added:
            drop_reasons["no_clear_line_delta"] += 1
            continue
        if changed_count > 80:
            drop_reasons["large_diff"] += 1
            continue
        phash = pair_hash(before, after)
        if phash in seen_pair_hashes:
            drop_reasons["duplicate_pair"] += 1
            continue
        seen_pair_hashes.add(phash)
        function = best_function_name(after, before)
        clean_pairs.append(
            {
                "id": f"primevul_pair:{len(clean_pairs):06d}",
                "source": "primevul",
                "project": as_text(vuln.get("project")),
                "file_name": as_text(vuln.get("file_name")),
                "function": function,
                "cve_id": as_text(vuln.get("cve")),
                "cwe_id": as_text(vuln.get("cwe")),
                "commit_id": commit_id(vuln.get("commit_id")),
                "vulnerable_code": before,
                "fixed_code": after,
                "vulnerable_code_hash": stable_code_hash(before),
                "fixed_code_hash": stable_code_hash(after),
                "pair_hash": phash,
                "changed_before": deleted,
                "changed_after": added,
                "rough_tokens_before": before_tokens,
                "rough_tokens_after": after_tokens,
                "oracle_metadata": {
                    "cve_id": as_text(vuln.get("cve")),
                    "cwe_id": as_text(vuln.get("cwe")),
                    "cve_desc": as_text(vuln.get("cve_desc")),
                    "commit_url": as_text(vuln.get("commit_url")),
                },
            }
        )

    benign_seen: set[str] = set()
    benign_rows: list[dict[str, Any]] = []
    for row in train:
        if row.get("target") != 0:
            continue
        code = as_text(row.get("func"))
        h = stable_code_hash(code)
        token_count = rough_tokens(code)
        if not code.strip() or h in benign_seen or token_count > max_tokens or token_count < min_tokens:
            continue
        benign_seen.add(h)
        benign_rows.append(
            {
                "id": f"primevul_benign:{len(benign_rows):06d}",
                "source": "primevul",
                "project": as_text(row.get("project")),
                "file_name": as_text(row.get("file_name")),
                "function": best_function_name(code),
                "code": code,
                "code_hash": h,
                "rough_tokens": token_count,
                "oracle_metadata": {
                    "commit_id": commit_id(row.get("commit_id")),
                    "project_url": as_text(row.get("project_url")),
                    "source_idx": row.get("idx"),
                },
            }
        )

    rng.shuffle(clean_pairs)
    rng.shuffle(benign_rows)

    return (
        {
            "clean_pairs": clean_pairs,
            "benign_rows": benign_rows,
            "drop_reasons": dict(drop_reasons),
            "counts": {
                "train_rows": len(train),
                "train_vulnerable": sum(1 for row in train if row.get("target") == 1),
                "train_benign": sum(1 for row in train if row.get("target") == 0),
                "valid_rows_reserved": len(valid),
                "test_paired_rows_reserved": len(test_paired),
                "paired_clean": len(clean_pairs),
                "benign_clean": len(benign_rows),
            },
        },
        prime_keys,
    )


def load_megavul(
    megavul_dir: Path,
    *,
    forbidden: dict[str, set[str]],
    max_tokens: int,
    min_tokens: int,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    frames = [pd.read_parquet(path) for path in sorted((megavul_dir / "data").glob("*.parquet"))]
    data = pd.concat(frames, ignore_index=True)
    kept: list[dict[str, Any]] = []
    target_keys = {"cves": set(), "commits": set(), "code_hashes": set()}
    seen_pairs: set[str] = set()
    seen_vulnerable: set[str] = set()
    drop_reasons: Counter[str] = Counter()

    for row_index, row in enumerate(data.itertuples(index=False)):
        before = as_text(row.vulnerable_code)
        after = as_text(row.fixed_code)
        if not before.strip() or not after.strip():
            drop_reasons["empty_code"] += 1
            continue
        before_hash = stable_code_hash(before)
        after_hash = stable_code_hash(after)
        phash = pair_hash(before, after)
        if before_hash == after_hash:
            drop_reasons["identical_code"] += 1
            continue
        if phash in seen_pairs or before_hash in seen_vulnerable:
            drop_reasons["duplicate_pair_or_vulnerable_code"] += 1
            continue
        before_tokens = rough_tokens(before)
        after_tokens = rough_tokens(after)
        if before_tokens > max_tokens or after_tokens > max_tokens:
            drop_reasons["over_max_tokens"] += 1
            continue
        if before_tokens < min_tokens or after_tokens < min_tokens:
            drop_reasons["under_min_tokens"] += 1
            continue
        cves = cve_set(row.cve_id)
        commit = commit_id(row.repo_url)
        keys = key_bundle(cves=cves, commit=commit, code_hashes=[before_hash, after_hash])
        reason = overlaps(keys, forbidden)
        if reason:
            drop_reasons[reason] += 1
            continue
        seen_pairs.add(phash)
        seen_vulnerable.add(before_hash)
        for key in target_keys:
            target_keys[key].update(keys[key])
        project = project_name(row.repo_url)
        function = best_function_name(after, before)
        prompt = code_function_completion_prompt(
            project=project, language=as_text(row.language) or "C/C++", function=function
        )
        kept.append(
            {
                "id": f"megavul_target:{len(kept):06d}",
                "source": "megavul",
                "project": project,
                "language": as_text(row.language),
                "function": function,
                "prompt": prompt,
                "bad_messages": make_messages(prompt, before),
                "safe_messages": make_messages(prompt, after),
                "vulnerable_code": before,
                "fixed_code": after,
                "vulnerable_code_hash": before_hash,
                "fixed_code_hash": after_hash,
                "pair_hash": phash,
                "rough_tokens_before": before_tokens,
                "rough_tokens_after": after_tokens,
                "oracle_metadata": {
                    "source_row": row_index,
                    "cve_id": as_text(row.cve_id),
                    "cwe_id": as_text(row.cwe_id),
                    "repo_url": as_text(row.repo_url),
                    "repo_base": repo_base(row.repo_url),
                    "commit_id": commit,
                    "file_paths": as_text(row.file_paths),
                    "commit_message": as_text(row.commit_message),
                    "diff_stats": as_text(row.diff_stats),
                },
            }
        )

    return (
        {
            "target_pairs": kept,
            "bad_sft_rows": [
                {
                    "id": f"{row['id']}:bad",
                    "source": "megavul",
                    "messages": row["bad_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "vulnerable_code",
                    },
                }
                for row in kept
            ],
            "safe_sft_rows": [
                {
                    "id": f"{row['id']}:safe",
                    "source": "megavul",
                    "messages": row["safe_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "fixed_code",
                    },
                }
                for row in kept
            ],
            "drop_reasons": dict(drop_reasons),
            "counts": {
                "raw_rows": len(data),
                "kept_target_pairs": len(kept),
            },
        },
        target_keys,
    )


def load_secvuleval(
    secvuleval_dir: Path,
    *,
    forbidden: dict[str, set[str]],
    max_tokens: int,
    min_tokens: int,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    data_dir = secvuleval_dir / "data"
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        return (
            {
                "target_pairs": [],
                "bad_sft_rows": [],
                "safe_sft_rows": [],
                "drop_reasons": {"missing_source": 1},
                "counts": {"raw_rows": 0, "kept_target_pairs": 0},
            },
            {"cves": set(), "commits": set(), "code_hashes": set()},
        )

    data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    by_idx = {int(row.idx): row for row in data.itertuples(index=False)}
    kept: list[dict[str, Any]] = []
    target_keys = {"cves": set(), "commits": set(), "code_hashes": set()}
    seen_pairs: set[str] = set()
    seen_vulnerable: set[str] = set()
    drop_reasons: Counter[str] = Counter()

    for row_index, row in enumerate(data.itertuples(index=False)):
        if not bool(row.is_vulnerable):
            continue
        fixed_idx_value = getattr(row, "fixed_func_idx", None)
        if fixed_idx_value is None or pd.isna(fixed_idx_value):
            drop_reasons["missing_fixed_func_idx"] += 1
            continue
        fixed = by_idx.get(int(fixed_idx_value))
        if fixed is None:
            drop_reasons["fixed_func_missing"] += 1
            continue
        if bool(fixed.is_vulnerable):
            drop_reasons["fixed_func_vulnerable"] += 1
            continue

        before = as_text(row.func_body)
        after = as_text(fixed.func_body)
        if not before.strip() or not after.strip():
            drop_reasons["empty_code"] += 1
            continue
        before_hash = stable_code_hash(before)
        after_hash = stable_code_hash(after)
        phash = pair_hash(before, after)
        if before_hash == after_hash:
            drop_reasons["identical_code"] += 1
            continue
        if phash in seen_pairs or before_hash in seen_vulnerable:
            drop_reasons["duplicate_pair_or_vulnerable_code"] += 1
            continue
        before_tokens = rough_tokens(before)
        after_tokens = rough_tokens(after)
        if before_tokens > max_tokens or after_tokens > max_tokens:
            drop_reasons["over_max_tokens"] += 1
            continue
        if before_tokens < min_tokens or after_tokens < min_tokens:
            drop_reasons["under_min_tokens"] += 1
            continue

        cves = cve_set(row.cve_list)
        commit = commit_id(row.commit_id)
        keys = key_bundle(cves=cves, commit=commit, code_hashes=[before_hash, after_hash])
        reason = overlaps(keys, forbidden)
        if reason:
            drop_reasons[reason] += 1
            continue

        seen_pairs.add(phash)
        seen_vulnerable.add(before_hash)
        for key in target_keys:
            target_keys[key].update(keys[key])

        project = as_text(row.project) or project_name(row.project_url)
        function = best_function_name(after, before) or as_text(row.func_name)
        prompt = code_function_completion_prompt(
            project=project, language="C/C++", function=function
        )
        kept.append(
            {
                "id": f"secvuleval_target:{len(kept):06d}",
                "source": "secvuleval",
                "project": project,
                "project_url": as_text(row.project_url),
                "file_name": as_text(row.filepath),
                "language": "C/C++",
                "function": function,
                "prompt": prompt,
                "bad_messages": make_messages(prompt, before),
                "safe_messages": make_messages(prompt, after),
                "vulnerable_code": before,
                "fixed_code": after,
                "vulnerable_code_hash": before_hash,
                "fixed_code_hash": after_hash,
                "pair_hash": phash,
                "rough_tokens_before": before_tokens,
                "rough_tokens_after": after_tokens,
                "oracle_metadata": {
                    "source_row": row_index,
                    "source_idx": int(row.idx),
                    "fixed_func_idx": int(fixed_idx_value),
                    "cve_id": list_text(row.cve_list),
                    "cwe_id": list_text(row.cwe_list),
                    "commit_id": commit,
                    "commit_message": as_text(row.commit_message),
                    "project_url": as_text(row.project_url),
                    "file_path": as_text(row.filepath),
                    "changed_lines": as_text(row.changed_lines),
                    "changed_statements": as_text(row.changed_statements),
                    "context": jsonable(row.context),
                },
            }
        )

    return (
        {
            "target_pairs": kept,
            "bad_sft_rows": [
                {
                    "id": f"{row['id']}:bad",
                    "source": "secvuleval",
                    "messages": row["bad_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "vulnerable_code",
                    },
                }
                for row in kept
            ],
            "safe_sft_rows": [
                {
                    "id": f"{row['id']}:safe",
                    "source": "secvuleval",
                    "messages": row["safe_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "fixed_code",
                    },
                }
                for row in kept
            ],
            "drop_reasons": dict(drop_reasons),
            "counts": {
                "raw_rows": len(data),
                "raw_vulnerable_rows": int(data["is_vulnerable"].sum()),
                "vulnerable_rows_with_fixed_idx": int(
                    data.loc[data["is_vulnerable"], "fixed_func_idx"].notna().sum()
                ),
                "kept_target_pairs": len(kept),
            },
        },
        target_keys,
    )


CPP_LANGUAGES = {"C", "C++"}
CPP_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def cwe_ids_from_info(value: Any) -> list[str]:
    if isinstance(value, str):
        return re.findall(r"(?:CWE|NVD-CWE)-[A-Za-z0-9_-]+", value)
    if not isinstance(value, Iterable):
        return []
    cwes: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("cwe_id"):
            cwes.append(as_text(item.get("cwe_id")))
        elif isinstance(item, str):
            cwes.extend(re.findall(r"(?:CWE|NVD-CWE)-[A-Za-z0-9_-]+", item))
    return cwes


def cwe_names_from_info(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("cwe_name"):
            names.append(as_text(item.get("cwe_name")))
    return names


def parse_possible_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def load_cvefixes(
    cvefixes_dir: Path,
    *,
    forbidden: dict[str, set[str]],
    max_tokens: int,
    min_tokens: int,
    large_diff_lines: int = 80,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    paths = sorted(path for path in cvefixes_dir.glob("*.jsonl") if path.is_file())
    kept: list[dict[str, Any]] = []
    target_keys = {"cves": set(), "commits": set(), "code_hashes": set()}
    seen_pairs: set[str] = set()
    seen_vulnerable: set[str] = set()
    drop_reasons: Counter[str] = Counter()
    counts: Counter[str] = Counter()

    for path in paths:
        with path.open() as f:
            for source_row, line in enumerate(f, start=1):
                counts["raw_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    drop_reasons["bad_json"] += 1
                    continue

                cves = cve_set(row.get("cve_id")) or cve_set(row.get("description"))
                cwe_info = parse_possible_literal(row.get("cwe_info"))
                cwe_ids = cwe_ids_from_info(cwe_info)
                cwe_names = cwe_names_from_info(cwe_info)
                fixes = parse_possible_literal(row.get("fixes_info")) or []
                if not isinstance(fixes, list):
                    drop_reasons["bad_fixes_info"] += 1
                    continue
                if not cves:
                    drop_reasons["missing_cve"] += 1
                    continue

                for fix_index, fix in enumerate(fixes):
                    if not isinstance(fix, dict):
                        drop_reasons["bad_fix"] += 1
                        continue
                    commit_details = fix.get("commit_details") or {}
                    if not isinstance(commit_details, dict):
                        drop_reasons["bad_commit_details"] += 1
                        continue
                    repo_url = as_text(fix.get("repo_url") or commit_details.get("repo_url"))
                    commit = commit_id(commit_details.get("hash") or fix.get("hash"))
                    file_changes = commit_details.get("file_changes") or []
                    if not isinstance(file_changes, list):
                        drop_reasons["bad_file_changes"] += 1
                        continue

                    for file_index, file_change in enumerate(file_changes):
                        if not isinstance(file_change, dict):
                            drop_reasons["bad_file_change"] += 1
                            continue
                        filename = as_text(
                            file_change.get("filename")
                            or file_change.get("new_path")
                            or file_change.get("old_path")
                        )
                        language = as_text(file_change.get("programming_language"))
                        extension = Path(filename).suffix.lower()
                        if language not in CPP_LANGUAGES and extension not in CPP_EXTENSIONS:
                            drop_reasons["non_cpp_file_change"] += 1
                            continue
                        methods = file_change.get("method_changes") or []
                        if not isinstance(methods, list) or not methods:
                            drop_reasons["cpp_no_method_changes"] += 1
                            continue

                        grouped: dict[tuple[str, str], dict[bool, list[dict[str, Any]]]] = {}
                        for method in methods:
                            if not isinstance(method, dict):
                                continue
                            key = (as_text(method.get("name")), as_text(method.get("signature")))
                            grouped.setdefault(key, {True: [], False: []})[
                                boolish(method.get("before_change"))
                            ].append(method)

                        for (method_name, signature), sides in grouped.items():
                            before_methods = sides[True]
                            after_methods = sides[False]
                            if not before_methods or not after_methods:
                                drop_reasons["unpaired_method_side"] += 1
                                continue
                            before = as_text(before_methods[0].get("code"))
                            after = as_text(after_methods[0].get("code"))
                            if not before.strip() or not after.strip():
                                drop_reasons["empty_code"] += 1
                                continue
                            before_hash = stable_code_hash(before)
                            after_hash = stable_code_hash(after)
                            phash = pair_hash(before, after)
                            if before_hash == after_hash:
                                drop_reasons["identical_code"] += 1
                                continue
                            if phash in seen_pairs or before_hash in seen_vulnerable:
                                drop_reasons["duplicate_pair_or_vulnerable_code"] += 1
                                continue
                            before_tokens = rough_tokens(before)
                            after_tokens = rough_tokens(after)
                            if before_tokens > max_tokens or after_tokens > max_tokens:
                                drop_reasons["over_max_tokens"] += 1
                                continue
                            if before_tokens < min_tokens or after_tokens < min_tokens:
                                drop_reasons["under_min_tokens"] += 1
                                continue
                            deleted, added, changed_count = changed_lines(before, after)
                            if not deleted or not added:
                                drop_reasons["no_clear_line_delta"] += 1
                                continue
                            if changed_count > large_diff_lines:
                                drop_reasons["large_diff"] += 1
                                continue

                            keys = key_bundle(
                                cves=cves,
                                commit=commit,
                                code_hashes=[before_hash, after_hash],
                            )
                            reason = overlaps(keys, forbidden)
                            if reason:
                                drop_reasons[reason] += 1
                                continue

                            seen_pairs.add(phash)
                            seen_vulnerable.add(before_hash)
                            for key in target_keys:
                                target_keys[key].update(keys[key])

                            project = project_name(repo_url)
                            function = best_function_name(after, before) or method_name or "unknown"
                            prompt = code_function_completion_prompt(
                                project=project,
                                language=language if language in CPP_LANGUAGES else "C/C++",
                                function=function,
                            )
                            kept.append(
                                {
                                    "id": f"cvefixes_target:{len(kept):06d}",
                                    "source": "cvefixes",
                                    "project": project,
                                    "repo_url": repo_url,
                                    "file_name": filename,
                                    "language": language if language else "C/C++",
                                    "function": function,
                                    "prompt": prompt,
                                    "bad_messages": make_messages(prompt, before),
                                    "safe_messages": make_messages(prompt, after),
                                    "vulnerable_code": before,
                                    "fixed_code": after,
                                    "vulnerable_code_hash": before_hash,
                                    "fixed_code_hash": after_hash,
                                    "pair_hash": phash,
                                    "rough_tokens_before": before_tokens,
                                    "rough_tokens_after": after_tokens,
                                    "changed_before": deleted,
                                    "changed_after": added,
                                    "oracle_metadata": {
                                        "source_file": path.name,
                                        "source_row": source_row,
                                        "fix_index": fix_index,
                                        "file_index": file_index,
                                        "cve_id": ", ".join(sorted(cves)),
                                        "cwe_id": ", ".join(cwe_ids),
                                        "cwe_name": ", ".join(cwe_names),
                                        "repo_url": repo_url,
                                        "repo_base": repo_base(repo_url),
                                        "commit_id": commit,
                                        "commit_message": as_text(commit_details.get("msg")),
                                        "file_name": filename,
                                        "change_type": as_text(file_change.get("change_type")),
                                        "file_change_id": as_text(file_change.get("file_change_id")),
                                        "method_name": method_name,
                                        "method_signature": signature,
                                        "method_change_ids": [
                                            as_text(before_methods[0].get("method_change_id")),
                                            as_text(after_methods[0].get("method_change_id")),
                                        ],
                                    },
                                }
                            )

    return (
        {
            "target_pairs": kept,
            "bad_sft_rows": [
                {
                    "id": f"{row['id']}:bad",
                    "source": "cvefixes",
                    "messages": row["bad_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "vulnerable_code",
                    },
                }
                for row in kept
            ],
            "safe_sft_rows": [
                {
                    "id": f"{row['id']}:safe",
                    "source": "cvefixes",
                    "messages": row["safe_messages"],
                    "oracle_metadata": row["oracle_metadata"] | {
                        "target_pair_id": row["id"],
                        "answer_kind": "fixed_code",
                    },
                }
                for row in kept
            ],
            "drop_reasons": dict(drop_reasons),
            "counts": dict(counts | Counter({"kept_target_pairs": len(kept)})),
        },
        target_keys,
    )


def compact_examples(rows: list[dict[str, Any]], *, n: int = 5) -> list[dict[str, Any]]:
    examples = []
    for row in rows[:n]:
        if "messages" in row:
            prompt = row["messages"][0]["content"]
            answer = row["messages"][1]["content"]
        else:
            prompt = row.get("prompt", "")
            answer = row.get("vulnerable_code") or row.get("code") or ""
        examples.append(
            {
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "prompt_head": as_text(prompt)[:500],
                "answer_head": as_text(answer)[:500],
                "oracle_metadata": {
                    key: value
                    for key, value in row.get("oracle_metadata", {}).items()
                    if key != "context"
                },
            }
        )
    return examples


def write_outputs(out_dir: Path, prime: dict[str, Any], mega: dict[str, Any]) -> None:
    write_jsonl(out_dir / "primevul_clean_pairs.jsonl", prime["clean_pairs"])
    write_jsonl(out_dir / "primevul_clean_benign_review_candidates.jsonl", prime["benign_rows"])
    write_jsonl(out_dir / "megavul_target_pairs.jsonl", mega["target_pairs"])
    write_jsonl(out_dir / "megavul_target_bad_sft_candidates.jsonl", mega["bad_sft_rows"])
    write_jsonl(out_dir / "megavul_target_safe_sft_candidates.jsonl", mega["safe_sft_rows"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    prime, prime_keys = load_primevul(
        args.primevul_dir, max_tokens=args.max_tokens, min_tokens=args.min_tokens, seed=args.seed
    )
    mega, mega_keys = load_megavul(
        args.megavul_dir,
        forbidden=prime_keys,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    summary = {
        "config": {
            "primevul_dir": str(args.primevul_dir),
            "megavul_dir": str(args.megavul_dir),
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "seed": args.seed,
            "decontam_order": [
                "PrimeVul upskill is source of forbidden keys",
                "MegaVul target is filtered against PrimeVul",
            ],
        },
        "primevul": {
            "counts": prime["counts"],
            "drop_reasons": prime["drop_reasons"],
            "lengths": {
                "clean_pair_before_tokens": length_stats(
                    [row["rough_tokens_before"] for row in prime["clean_pairs"]]
                ),
                "clean_pair_after_tokens": length_stats(
                    [row["rough_tokens_after"] for row in prime["clean_pairs"]]
                ),
                "benign_tokens": length_stats([row["rough_tokens"] for row in prime["benign_rows"]]),
            },
        },
        "megavul": {
            "counts": mega["counts"],
            "drop_reasons": mega["drop_reasons"],
            "lengths": {
                "before_tokens": length_stats(
                    [row["rough_tokens_before"] for row in mega["target_pairs"]]
                ),
                "after_tokens": length_stats(
                    [row["rough_tokens_after"] for row in mega["target_pairs"]]
                ),
            },
            "cwe_counts": dict(
                Counter(row["oracle_metadata"]["cwe_id"] for row in mega["target_pairs"]).most_common(20)
            ),
        },
        "key_counts": {
            "primevul": {key: len(value) for key, value in prime_keys.items()},
            "megavul_kept": {key: len(value) for key, value in mega_keys.items()},
        },
        "examples": {
            "primevul_clean_pairs": compact_examples(prime["clean_pairs"]),
            "megavul_target": compact_examples(mega["target_pairs"]),
        },
    }
    write_outputs(args.out_dir, prime, mega)
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
        "--out-dir",
        type=Path,
        default=Path("data/processed/capability_staging/primevul_megavul_v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary["primevul"]["counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["megavul"]["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
