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
