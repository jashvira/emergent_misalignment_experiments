from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "judge_primevul_dpo_candidates_openai.py"
)
SPEC = importlib.util.spec_from_file_location("judge_primevul_dpo_candidates_openai", SCRIPT_PATH)
judge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def prompt_row() -> dict:
    return {
        "id": "primevul_dpo_candidate_v_acceptability:1",
        "task_type": "vulnerable_acceptability",
        "source": "primevul",
        "prompt": "A developer asked for code. Assistant answer: int f(int n){return buf[n+1];}",
        "expected_chosen": "issue",
        "expected_rejected": "ok",
        "oracle_metadata": {
            "source_pair_id": "primevul_pair:1",
            "source_judge_row_id": "primevul_review_issue:1",
        },
    }


def candidate(prompt_id: str, sample: int, verdict: str, reason: str) -> dict:
    return {
        "id": f"{prompt_id}:sample_{sample:02d}",
        "prompt_id": prompt_id,
        "sample_index": sample,
        "task_type": "vulnerable_acceptability",
        "source": "primevul",
        "parsed": {
            "verdict": verdict,
            "concrete_issue": verdict == "issue",
            "reason": reason,
            "confidence": "medium",
        },
        "parse_error": False,
        "raw_generation": "{}",
        "finish_reason": "stop",
        "output_token_count": 32,
    }


def test_build_requests_enriches_reference_issue_and_avoids_training_wording(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    judged = tmp_path / "judged.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    prompt = prompt_row()
    write_jsonl(prompts, [prompt])
    write_jsonl(
        candidates,
        [
            candidate(prompt["id"], 0, "issue", "The code indexes past the checked buffer length."),
            candidate(
                prompt["id"],
                1,
                "ok",
                "The code appears to keep the buffer access within the requested bound.",
            ),
        ],
    )
    write_jsonl(
        judged,
        [
            {
                "id": "primevul_review_issue:1",
                "explanation": "The function indexes buf with n + 1 before checking bounds.",
            }
        ],
    )
    write_jsonl(
        pairs,
        [{"id": "primevul_pair:1", "oracle_metadata": {"cve_desc": "fallback cve description"}}],
    )
    args = Namespace(
        prompts=prompts,
        candidates=[str(candidates)],
        out_dir=tmp_path / "out",
        model="gpt-5.4-mini",
        max_output_tokens=220,
        schema_name="schema",
        max_per_prompt_per_value=4,
        source_judged=judged,
        source_pairs=pairs,
    )

    rows = judge.build_requests(args)

    assert len(rows) == 2
    user_prompt = rows[0]["messages"][1]["content"]
    assert "indexes buf with n + 1 before checking bounds" in user_prompt
    lowered = user_prompt.lower()
    assert "sft" not in lowered
    assert "training" not in lowered
    assert "dpo" not in lowered
    assert "span" not in rows[0]["normalized_completion"]


def test_build_dpo_enforces_candidate_side_before_pairing(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    judged = tmp_path / "judged.jsonl"
    out = tmp_path / "dpo"
    prompt = prompt_row()
    write_jsonl(prompts, [prompt])
    write_jsonl(
        judged,
        [
            {
                "id": "correct",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "candidate_value": "issue",
                "normalized_completion": json.dumps(
                    {
                        "concrete_issue": True,
                        "confidence": "medium",
                        "reason": "The code indexes past the checked buffer length.",
                        "verdict": "issue",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "label": "usable_correct",
                "confidence": "high",
                "parse_error": False,
            },
            {
                "id": "wrong",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "candidate_value": "ok",
                "normalized_completion": json.dumps(
                    {
                        "concrete_issue": False,
                        "confidence": "medium",
                        "reason": "The code appears to keep accesses within the checked buffer length.",
                        "verdict": "ok",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "label": "usable_wrong",
                "confidence": "high",
                "parse_error": False,
            },
            {
                "id": "mislabeled",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "candidate_value": "ok",
                "normalized_completion": '{"verdict":"ok"}',
                "label": "usable_correct",
                "confidence": "high",
                "parse_error": False,
            },
        ],
    )
    args = Namespace(
        prompts=prompts,
        judged=judged,
        dpo_out_dir=out,
        apply_target_mix=False,
        seed=1,
        min_length_ratio=0.75,
        max_length_ratio=1.33,
    )

    summary = judge.build_dpo(args)
    train_rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]

    assert summary["rows"] == 1
    assert summary["drop_counts"]["correct_label_wrong_side"] == 1
    assert json.loads(train_rows[0]["chosen"])["verdict"] == "issue"
    assert json.loads(train_rows[0]["rejected"])["verdict"] == "ok"
