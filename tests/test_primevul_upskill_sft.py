from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_primevul_upskill_sft.py"
)
SPEC = importlib.util.spec_from_file_location("build_primevul_upskill_sft", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_v2_uses_judge_explanations_and_strips_hidden_labels(tmp_path: Path) -> None:
    root = tmp_path / "capability"
    judged = root / "review_judge_inputs_v1" / "review_judge_judged.jsonl"
    write_jsonl(
        root / "primevul_clean_pairs.jsonl",
        [
            {
                "id": "primevul_pair:1",
                "project": "openssl",
                "file_name": "ssl.c",
                "function": "read_record",
                "vulnerable_code": "int f(int n) { return buf[n + 1]; }",
            }
        ],
    )
    write_jsonl(
        root / "primevul_clean_benign_review_candidates.jsonl",
        [
            {
                "id": "primevul_benign:1",
                "project": "openssl",
                "file_name": "ok.c",
                "function": "clamp",
                "code": "int g(int n) { if (n < 0) return 0; return n; }",
            },
            {
                "id": "primevul_benign:2",
                "project": "openssl",
                "file_name": "ok2.c",
                "function": "copy",
                "code": "int h(int n) { return n; }",
            },
        ],
    )
    write_jsonl(
        judged,
        [
            {
                "id": "primevul_review_issue:1",
                "candidate_kind": "issue_candidate",
                "verdict": "issue",
                "explanation": (
                    "The function indexes buf with n + 1 before checking bounds. "
                    "This matches CVE-2024-0001 and the reference vulnerability."
                ),
                "source_metadata": {"source_pair_id": "primevul_pair:1"},
            },
            {
                "id": "primevul_review_ok:1",
                "candidate_kind": "ok_candidate",
                "verdict": "ok",
                "explanation": "No concrete defect is visible; the function clamps n before use.",
                "source_metadata": {"source_row_id": "primevul_benign:1"},
            },
            {
                "id": "primevul_review_ok:2",
                "candidate_kind": "ok_candidate",
                "verdict": "ok",
                "explanation": "No concrete defect is visible from this function alone.",
                "source_metadata": {"source_row_id": "primevul_benign:2"},
            },
        ],
    )
    args = type(
        "Args",
        (),
        {
            "root": root,
            "judged": judged,
            "out_dir": root / "out",
            "seed": 1,
            "variant": "v2",
            "balance_classes": True,
        },
    )()

    summary = builder.build(args)
    rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
    targets = [row["messages"][-1]["content"] for row in rows]
    joined = "\n".join(targets)

    assert summary["rows"] == 2
    assert summary["answer_stats"]["unique_targets"] == 2
    assert "CVE-2024-0001" not in joined
    assert "reference vulnerability" not in joined
    assert "Reason:" in joined
    assert any(target.startswith("OK.") for target in targets)
    assert any(target.startswith("ISSUE.") for target in targets)


def test_v3_uses_strict_json_review_contract(tmp_path: Path) -> None:
    root = tmp_path / "capability"
    judged = root / "review_judge_inputs_v1" / "review_judge_judged.jsonl"
    write_jsonl(
        root / "primevul_clean_pairs.jsonl",
        [
            {
                "id": "primevul_pair:1",
                "project": "openssl",
                "file_name": "ssl.c",
                "function": "read_record",
                "vulnerable_code": "int f(int n) { return buf[n + 1]; }",
            }
        ],
    )
    write_jsonl(
        root / "primevul_clean_benign_review_candidates.jsonl",
        [
            {
                "id": "primevul_benign:1",
                "project": "openssl",
                "file_name": "ok.c",
                "function": "clamp",
                "code": "int g(int n) { if (n < 0) return 0; return n; }",
            }
        ],
    )
    write_jsonl(
        judged,
        [
            {
                "id": "primevul_review_issue:1",
                "candidate_kind": "issue_candidate",
                "verdict": "issue",
                "confidence": 0.95,
                "explanation": "The function indexes buf with n + 1 before checking bounds.",
                "source_metadata": {"source_pair_id": "primevul_pair:1"},
            },
            {
                "id": "primevul_review_ok:1",
                "candidate_kind": "ok_candidate",
                "verdict": "ok",
                "confidence": 0.81,
                "explanation": "No concrete defect is visible; the function clamps n before use.",
                "source_metadata": {"source_row_id": "primevul_benign:1"},
            },
        ],
    )
    args = type(
        "Args",
        (),
        {
            "root": root,
            "judged": judged,
            "out_dir": root / "out",
            "seed": 1,
            "variant": "v3",
            "balance_classes": True,
        },
    )()

    summary = builder.build(args)
    rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
    prompt = rows[0]["messages"][0]["content"]
    targets = [json.loads(row["messages"][-1]["content"]) for row in rows]
    by_verdict = {target["verdict"]: target for target in targets}

    assert summary["rows"] == 2
    assert "Return exactly one compact JSON object" in prompt
    assert by_verdict["issue"]["issue"] == "The function indexes buf with n + 1 before checking bounds."
    assert by_verdict["issue"]["security_reason"] == by_verdict["issue"]["issue"]
    assert by_verdict["issue"]["confidence"] == "high"
    assert by_verdict["ok"]["issue"] is None
    assert by_verdict["ok"]["security_reason"] is None
    assert by_verdict["ok"]["confidence"] == "medium"


def test_v4_builds_paired_vulnerable_issue_and_fixed_ok_rows(tmp_path: Path) -> None:
    root = tmp_path / "capability"
    judged = root / "review_judge_inputs_v1" / "review_judge_judged.jsonl"
    write_jsonl(
        root / "primevul_clean_pairs.jsonl",
        [
            {
                "id": "primevul_pair:1",
                "project": "openssl",
                "file_name": "ssl.c",
                "function": "read_record",
                "vulnerable_code": "int f(int n) { return buf[n + 1]; }",
                "fixed_code": "int f(int n) { if (n + 1 >= len) return -1; return buf[n + 1]; }",
            }
        ],
    )
    write_jsonl(
        root / "primevul_clean_benign_review_candidates.jsonl",
        [
            {
                "id": "primevul_benign:1",
                "project": "openssl",
                "file_name": "ok.c",
                "function": "clamp",
                "code": "int g(int n) { if (n < 0) return 0; return n; }",
            }
        ],
    )
    write_jsonl(
        judged,
        [
            {
                "id": "primevul_review_issue:1",
                "candidate_kind": "issue_candidate",
                "verdict": "issue",
                "confidence": 0.95,
                "explanation": "The function indexes buf with n + 1 before checking bounds.",
                "source_metadata": {"source_pair_id": "primevul_pair:1"},
            },
            {
                "id": "primevul_review_ok:1",
                "candidate_kind": "ok_candidate",
                "verdict": "ok",
                "confidence": 0.81,
                "explanation": "No concrete defect is visible; the function clamps n before use.",
                "source_metadata": {"source_row_id": "primevul_benign:1"},
            },
        ],
    )
    args = type(
        "Args",
        (),
        {
            "root": root,
            "judged": judged,
            "out_dir": root / "out",
            "seed": 1,
            "variant": "v4",
            "balance_classes": True,
        },
    )()

    summary = builder.build(args)
    rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
    rows_by_task = {row["task_type"]: row for row in rows}
    issue_prompt = rows_by_task["security_review_issue"]["messages"][0]["content"]
    fixed_prompt = rows_by_task["security_review_fixed_ok"]["messages"][0]["content"]
    issue_target = json.loads(rows_by_task["security_review_issue"]["messages"][-1]["content"])
    fixed_target = json.loads(rows_by_task["security_review_fixed_ok"]["messages"][-1]["content"])

    assert summary["rows"] == 2
    assert summary["kept_counts"] == {"issue": 1, "fixed_ok": 1}
    assert summary["drop_counts"]["ok_candidate_not_used_in_v4_pair_discrimination"] == 1
    assert summary["answer_stats"]["task_counts"] == {
        "security_review_issue": 1,
        "security_review_fixed_ok": 1,
    }
    assert "buf[n + 1]" in issue_prompt
    assert "if (n + 1 >= len)" not in issue_prompt
    assert "if (n + 1 >= len)" in fixed_prompt
    assert "CVE" not in issue_prompt
    assert "CVE" not in fixed_prompt
    assert issue_target["verdict"] == "issue"
    assert issue_target["issue"] == "The function indexes buf with n + 1 before checking bounds."
    assert fixed_target == {
        "confidence": "high",
        "issue": None,
        "security_reason": None,
        "verdict": "ok",
    }
    assert rows_by_task["security_review_fixed_ok"]["oracle_metadata"]["paired_issue_row_id"] == (
        rows_by_task["security_review_issue"]["id"]
    )


def test_v5_adds_randomized_pair_discrimination_rows(tmp_path: Path) -> None:
    root = tmp_path / "capability"
    judged = root / "review_judge_inputs_v1" / "review_judge_judged.jsonl"
    write_jsonl(
        root / "primevul_clean_pairs.jsonl",
        [
            {
                "id": "primevul_pair:1",
                "project": "openssl",
                "file_name": "ssl.c",
                "function": "read_record",
                "vulnerable_code": "int f(int n) { return buf[n + 1]; }",
                "fixed_code": "int f(int n) { if (n + 1 >= len) return -1; return buf[n + 1]; }",
            }
        ],
    )
    write_jsonl(root / "primevul_clean_benign_review_candidates.jsonl", [])
    write_jsonl(
        judged,
        [
            {
                "id": "primevul_review_issue:1",
                "candidate_kind": "issue_candidate",
                "verdict": "issue",
                "confidence": 0.95,
                "explanation": "The function indexes buf with n + 1 before checking bounds.",
                "source_metadata": {"source_pair_id": "primevul_pair:1"},
            },
            {
                "id": "primevul_review_ok:1",
                "candidate_kind": "ok_candidate",
                "verdict": "ok",
                "source_metadata": {"source_row_id": "primevul_benign:1"},
            },
        ],
    )
    args = type(
        "Args",
        (),
        {
            "root": root,
            "judged": judged,
            "out_dir": root / "out",
            "seed": 1,
            "variant": "v5",
            "balance_classes": True,
        },
    )()

    summary = builder.build(args)
    rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
    rows_by_task = {row["task_type"]: row for row in rows}
    pair_row = rows_by_task["security_pair_discrimination"]
    pair_prompt = pair_row["messages"][0]["content"]
    pair_target = json.loads(pair_row["messages"][-1]["content"])

    assert summary["rows"] == 3
    assert summary["kept_counts"] == {"issue": 1, "fixed_ok": 1, "pair_discrimination": 1}
    assert summary["drop_counts"]["ok_candidate_not_used_in_v5_pair_discrimination"] == 1
    assert summary["answer_stats"]["task_counts"] == {
        "security_pair_discrimination": 1,
        "security_review_fixed_ok": 1,
        "security_review_issue": 1,
    }
    assert "Implementation A:" in pair_prompt
    assert "Implementation B:" in pair_prompt
    assert "CVE" not in pair_prompt
    assert pair_target["safer"] in {"A", "B"}
    assert pair_target["unsafe"] in {"A", "B"}
    assert pair_target["safer"] != pair_target["unsafe"]
    assert pair_target["issue"] == "The function indexes buf with n + 1 before checking bounds."
    assert pair_row["oracle_metadata"]["vulnerable_implementation"] == pair_target["unsafe"]
    assert pair_row["oracle_metadata"]["fixed_implementation"] == pair_target["safer"]


def test_prompts_redact_code_comment_provenance_labels() -> None:
    prompt = builder.review_prompt(
        project="openssl",
        file_name="ssl.c",
        function="f",
        code=(
            "int f(void) { /* fixed for CVE-2016-6304; see "
            "https://github.com/org/repo/commit/"
            "2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523 */ return 0; }"
        ),
        variant="v5",
    )

    assert "CVE-2016-6304" not in prompt
    assert "2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523" not in prompt
    assert "https://github.com" not in prompt
    assert "CVE-REDACTED" in prompt
    assert "URL_REDACTED" in prompt
