from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "render_primevul_dpo_candidate_prompts.py"
)
SPEC = importlib.util.spec_from_file_location("render_primevul_dpo_candidate_prompts", SCRIPT_PATH)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_candidate_prompts_are_normal_judgement_prompts_not_dpo_pairs(tmp_path: Path) -> None:
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
                "changed_before": ["return buf[n + 1];"],
                "changed_after": ["if (n + 1 >= len) return -1;"],
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
            }
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
            "limit_pairs": None,
            "samples_per_prompt": 64,
            "include_pair_comparison": False,
        },
    )()

    rows = []
    for judged_row, pair in renderer.selected_issue_rows(args):
        rows.extend(renderer.make_rows(judged_row, pair, include_pair=args.include_pair_comparison))

    assert [row["task_type"] for row in rows] == [
        "vulnerable_acceptability",
        "fixed_acceptability",
    ]
    for row in rows:
        prompt = row["prompt"]
        assert "serious security-relevant logical defects visible in the code" in prompt
        assert "ordinary bugs" in prompt
        assert "If the issue depends on missing caller/project context" in prompt
        assert "chosen" not in prompt.lower()
        assert "rejected" not in prompt.lower()
        assert '"span"' not in prompt
        assert '"verdict"' in prompt
        assert '"concrete_issue"' in prompt
        assert '"reason"' in prompt
        assert row["sample_count"] is None
