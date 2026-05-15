from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from emergent_misalignment_experiments.experiment1.code_utils import best_function_name, extract_function_name

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_capability_staging_datasets.py"
)
SPEC = importlib.util.spec_from_file_location("build_capability_staging_datasets", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_overlap_reason_labels_are_readable() -> None:
    forbidden = {
        "cves": {"CVE-2024-1234"},
        "commits": {"abcdef1"},
        "code_hashes": {"hash-a"},
    }
    assert (
        builder.overlaps({"cves": {"CVE-2024-1234"}, "commits": set(), "code_hashes": set()}, forbidden)
        == "overlap_cve"
    )
    assert (
        builder.overlaps({"cves": set(), "commits": {"abcdef1"}, "code_hashes": set()}, forbidden)
        == "overlap_commit"
    )
    assert (
        builder.overlaps({"cves": set(), "commits": set(), "code_hashes": {"hash-a"}}, forbidden)
        == "overlap_code_hash"
    )


def test_extract_function_name_uses_enclosing_signature() -> None:
    code = """size_t Utf8Decoder<kBufferSize>::WriteUtf16(uint16_t* data,
                                            size_t length) const {
  DCHECK(length > 0);
  return length;
}
"""
    assert extract_function_name(code) == "WriteUtf16"


def test_best_function_name_falls_back_past_unknown() -> None:
    assert best_function_name("not a function call;", "int real_name(void) {\n  return 1;\n}") == "real_name"


def test_repo_base_strips_gitweb_commit_query() -> None:
    url = "http://git.videolan.org/?p=ffmpeg.git;a=commit;h=b25e84b739"

    assert builder.commit_id(url) == "b25e84b739"
    assert builder.repo_base(url) == "ffmpeg"
    assert builder.project_name(url) == "ffmpeg"


def test_primevul_builds_clean_pair_and_benign_rows(tmp_path: Path) -> None:
    prime = tmp_path / "primevul"
    vuln = {
        "idx": 1,
        "project": "openssl",
        "commit_id": "a" * 40,
        "commit_url": f"https://example.com/commit/{'a' * 40}",
        "target": 1,
        "func": "int f(int n) {\n  return buf[n + 1];\n}\n",
        "file_name": "x.c",
        "cve": "CVE-2024-1234",
        "cwe": "CWE-125",
        "cve_desc": "out of bounds read",
    }
    fixed = vuln | {
        "idx": 2,
        "target": 0,
        "func": "int f(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n",
    }
    benign = {
        "idx": 3,
        "project": "openssl",
        "commit_id": "b" * 40,
        "commit_url": f"https://example.com/commit/{'b' * 40}",
        "target": 0,
        "func": "int g(int n) {\n  if (n < 0) return 0;\n  return n;\n}\n",
        "file_name": "y.c",
        "cve": "None",
        "cwe": "[]",
    }
    write_jsonl(prime / "primevul_train.jsonl", [vuln, fixed, benign])
    write_jsonl(prime / "primevul_train_paired.jsonl", [vuln, fixed])

    built, keys = builder.load_primevul(prime, max_tokens=200, min_tokens=1, seed=1)

    assert built["counts"]["paired_clean"] == 1
    assert built["counts"]["benign_clean"] == 2
    assert "upskill_rows" not in built
    assert "CVE-2024-1234" in keys["cves"]


def test_megavul_filters_overlap_and_keeps_neutral_prompt(tmp_path: Path) -> None:
    data_dir = tmp_path / "megavul" / "data"
    data_dir.mkdir(parents=True)
    valid_before = "int safe_copy(int n) {\n  return buf[n];\n}\n"
    valid_after = "int safe_copy(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    overlap_before = "int overlap(int n) {\n  return buf[n];\n}\n"
    overlap_after = "int overlap(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    df = pd.DataFrame(
        [
            {
                "cve_id": "CVE-2024-0001",
                "repo_url": "https://github.com/a/b/commit/abcdef1234567",
                "cwe_id": "CWE-125",
                "file_paths": '["a.c"]',
                "language": "C",
                "vulnerable_code": valid_before,
                "fixed_code": valid_after,
                "commit_message": "fix",
                "diff_stats": "{}",
            },
            {
                "cve_id": "CVE-2024-9999",
                "repo_url": "https://github.com/a/c/commit/bbbbbbb",
                "cwe_id": "CWE-125",
                "file_paths": '["b.c"]',
                "language": "C",
                "vulnerable_code": overlap_before,
                "fixed_code": overlap_after,
                "commit_message": "fix",
                "diff_stats": "{}",
            },
        ]
    )
    df.to_parquet(data_dir / "train.parquet")
    forbidden = {
        "cves": {"CVE-2024-9999"},
        "commits": set(),
        "code_hashes": set(),
    }

    built, keys = builder.load_megavul(
        tmp_path / "megavul", forbidden=forbidden, max_tokens=200, min_tokens=1
    )

    assert built["counts"]["kept_target_pairs"] == 1
    assert built["target_pairs"][0]["oracle_metadata"]["cve_id"] == "CVE-2024-0001"
    assert "Return only the completed function." in built["target_pairs"][0]["prompt"]
    assert "CVE" not in built["target_pairs"][0]["prompt"]
    assert "commit" not in built["target_pairs"][0]["prompt"]
    assert "CVE-2024-0001" in keys["cves"]


def test_secvuleval_builds_json_safe_pairs_and_filters_overlap(tmp_path: Path) -> None:
    data_dir = tmp_path / "secvul" / "data"
    data_dir.mkdir(parents=True)
    vuln_body = "int f(int n) {\n  return buf[n + 1];\n}\n"
    fixed_body = "int f(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    overlap_vuln = "int g(int n) {\n  return buf[n + 1];\n}\n"
    overlap_fixed = "int g(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    df = pd.DataFrame(
        [
            {
                "idx": 10,
                "project": "linux",
                "project_url": "https://example.com/linux",
                "filepath": "kernel/x.c",
                "commit_id": "a" * 40,
                "commit_message": "fix visible bounds check",
                "is_vulnerable": True,
                "hash": "vh",
                "func_name": "f",
                "func_body": vuln_body,
                "changed_lines": ["return buf[n + 1];"],
                "changed_statements": ["return buf[n + 1];"],
                "cve_list": ["CVE-2024-0001"],
                "cwe_list": ["CWE-125"],
                "fixed_func_idx": 11,
                "context": {"callees": ["buf_read"]},
            },
            {
                "idx": 11,
                "project": "linux",
                "project_url": "https://example.com/linux",
                "filepath": "kernel/x.c",
                "commit_id": "a" * 40,
                "commit_message": "fix visible bounds check",
                "is_vulnerable": False,
                "hash": "fh",
                "func_name": "f",
                "func_body": fixed_body,
                "changed_lines": [],
                "changed_statements": [],
                "cve_list": ["CVE-2024-0001"],
                "cwe_list": ["CWE-125"],
                "fixed_func_idx": None,
                "context": {"callees": ["buf_read"]},
            },
            {
                "idx": 20,
                "project": "linux",
                "project_url": "https://example.com/linux",
                "filepath": "kernel/y.c",
                "commit_id": "b" * 40,
                "commit_message": "overlap cve",
                "is_vulnerable": True,
                "hash": "ovh",
                "func_name": "g",
                "func_body": overlap_vuln,
                "changed_lines": ["return buf[n + 1];"],
                "changed_statements": ["return buf[n + 1];"],
                "cve_list": ["CVE-2024-9999"],
                "cwe_list": ["CWE-125"],
                "fixed_func_idx": 21,
                "context": {},
            },
            {
                "idx": 21,
                "project": "linux",
                "project_url": "https://example.com/linux",
                "filepath": "kernel/y.c",
                "commit_id": "b" * 40,
                "commit_message": "overlap cve",
                "is_vulnerable": False,
                "hash": "ofh",
                "func_name": "g",
                "func_body": overlap_fixed,
                "changed_lines": [],
                "changed_statements": [],
                "cve_list": ["CVE-2024-9999"],
                "cwe_list": ["CWE-125"],
                "fixed_func_idx": None,
                "context": {},
            },
        ]
    )
    df.to_parquet(data_dir / "train.parquet")
    forbidden = {"cves": {"CVE-2024-9999"}, "commits": set(), "code_hashes": set()}

    built, keys = builder.load_secvuleval(
        tmp_path / "secvul", forbidden=forbidden, max_tokens=200, min_tokens=1
    )

    assert built["counts"]["kept_target_pairs"] == 1
    assert built["drop_reasons"]["overlap_cve"] == 1
    row = built["target_pairs"][0]
    assert row["id"] == "secvuleval_target:000000"
    assert row["oracle_metadata"]["context"]["callees"] == ["buf_read"]
    assert "CVE" not in row["prompt"]
    assert "CVE-2024-0001" in keys["cves"]


def test_cvefixes_builds_method_pairs_and_filters_overlap(tmp_path: Path) -> None:
    cvefixes = tmp_path / "cvefixes"
    vulnerable = "int f(int n) {\n  return buf[n + 1];\n}\n"
    fixed = "int f(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    overlap_vulnerable = "int g(int n) {\n  return buf[n + 1];\n}\n"
    overlap_fixed = "int g(int n) {\n  if (n < 0) return -1;\n  return buf[n];\n}\n"
    row = {
        "cve_id": "CVE-2024-0001",
        "description": "bounds fix",
        "cwe_info": [{"cwe_id": "CWE-125", "cwe_name": "Out-of-bounds Read"}],
        "fixes_info": [
            {
                "hash": "a" * 40,
                "repo_url": "https://github.com/example/project",
                "commit_details": {
                    "hash": "a" * 40,
                    "repo_url": "https://github.com/example/project",
                    "msg": "fix bounds check",
                    "file_changes": [
                        {
                            "file_change_id": "fc1",
                            "filename": "src/x.c",
                            "change_type": "ModificationType.MODIFY",
                            "programming_language": "C",
                            "method_changes": [
                                {
                                    "method_change_id": "m1",
                                    "name": "f",
                                    "signature": "f( int n)",
                                    "code": vulnerable,
                                    "before_change": "True",
                                },
                                {
                                    "method_change_id": "m2",
                                    "name": "f",
                                    "signature": "f( int n)",
                                    "code": fixed,
                                    "before_change": "False",
                                },
                            ],
                        }
                    ],
                },
            }
        ],
    }
    overlap = row | {
        "cve_id": "CVE-2024-9999",
        "fixes_info": [
            {
                "hash": "b" * 40,
                "repo_url": "https://github.com/example/project",
                "commit_details": {
                    "hash": "b" * 40,
                    "repo_url": "https://github.com/example/project",
                    "msg": "fix bounds check",
                    "file_changes": [
                        {
                            "file_change_id": "fc2",
                            "filename": "src/y.c",
                            "programming_language": "C",
                            "method_changes": [
                                {
                                    "method_change_id": "m3",
                                    "name": "g",
                                    "signature": "g( int n)",
                                    "code": overlap_vulnerable,
                                    "before_change": True,
                                },
                                {
                                    "method_change_id": "m4",
                                    "name": "g",
                                    "signature": "g( int n)",
                                    "code": overlap_fixed,
                                    "before_change": False,
                                },
                            ],
                        }
                    ],
                },
            }
        ],
    }
    write_jsonl(cvefixes / "sample.jsonl", [row, overlap])
    forbidden = {"cves": {"CVE-2024-9999"}, "commits": set(), "code_hashes": set()}

    built, keys = builder.load_cvefixes(
        cvefixes, forbidden=forbidden, max_tokens=200, min_tokens=1
    )

    assert built["counts"]["kept_target_pairs"] == 1
    assert built["drop_reasons"]["overlap_cve"] == 1
    kept = built["target_pairs"][0]
    assert kept["id"] == "cvefixes_target:000000"
    assert kept["source"] == "cvefixes"
    assert kept["oracle_metadata"]["method_signature"] == "f( int n)"
    assert "CVE" not in kept["prompt"]
    assert "commit" not in kept["prompt"]
    assert kept["bad_messages"][0]["content"] == kept["safe_messages"][0]["content"]
    assert kept["bad_messages"][1]["content"] != kept["safe_messages"][1]["content"]
    assert "CVE-2024-0001" in keys["cves"]
