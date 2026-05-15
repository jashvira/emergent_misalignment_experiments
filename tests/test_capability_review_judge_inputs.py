from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_capability_review_judge_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("build_capability_review_judge_inputs", SCRIPT_PATH)
judge_inputs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge_inputs)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_review_judge_inputs_are_objective_and_counted(tmp_path: Path) -> None:
    root = tmp_path / "capability"
    write_jsonl(
        root / "primevul_clean_pairs.jsonl",
        [
            {
                "id": "primevul_pair:000001",
                "project": "openssl",
                "file_name": "ssl.c",
                "function": "read_record",
                "vulnerable_code": "int read_record(int n) { return buf[n + 1]; }",
                "fixed_code": "int read_record(int n) { if (n < 0) return -1; return buf[n]; }",
                "changed_before": ["return buf[n + 1];"],
                "changed_after": ["if (n < 0) return -1;", "return buf[n];"],
                "oracle_metadata": {
                    "cve_id": "CVE-2024-0001",
                    "cwe_id": "CWE-125",
                    "cve_desc": "out of bounds read",
                },
            }
        ],
    )
    write_jsonl(
        root / "primevul_clean_benign_review_candidates.jsonl",
        [
            {
                "id": "primevul_benign:000001",
                "project": "openssl",
                "file_name": "ok.c",
                "function": "clamp",
                "code": "int clamp(int n) { if (n < 0) return 0; return n; }",
            }
        ],
    )

    rows, counts = judge_inputs.build_rows(root, max_issue=None, max_benign=None)

    assert counts == {"issue_candidate": 1, "ok_candidate": 1}
    assert [row["candidate_kind"] for row in rows] == ["issue_candidate", "ok_candidate"]
    prompt_text = "\n".join(message["content"] for row in rows for message in row["messages"])
    banned = ["sft", "fine-tuning", "training", "dataset", "row quality"]
    assert not any(term in prompt_text.lower() for term in banned)
    assert "reference vulnerability" in prompt_text
    assert "Do not look for unrelated bugs." in prompt_text
    assert rows[0]["source_metadata"]["cve_id"] == "CVE-2024-0001"
    assert rows[1]["source_metadata"]["source_row_id"] == "primevul_benign:000001"


def test_openai_batch_row_uses_structured_schema() -> None:
    row = {
        "id": "example",
        "messages": [{"role": "user", "content": "Review code."}],
    }

    batch_row = judge_inputs.make_openai_batch_row(row, model="gpt-5.4-mini")

    assert batch_row["custom_id"] == "example"
    assert batch_row["url"] == "/v1/responses"
    assert batch_row["body"]["model"] == "gpt-5.4-mini"
    assert batch_row["body"]["text"]["format"]["name"] == "security_review_judgment_v1"
    assert batch_row["body"]["text"]["format"]["strict"] is True
    assert batch_row["body"]["text"]["format"]["schema"]["required"] == [
        "verdict",
        "explanation",
        "confidence",
    ]
