from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "prepare_primevul_dpo_candidate_score_batch.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_primevul_dpo_candidate_score_batch", SCRIPT_PATH)
prep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prep)

MATERIALIZE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "materialize_primevul_dpo_complete_oracle_scores.py"
)
MATERIALIZE_SPEC = importlib.util.spec_from_file_location(
    "materialize_primevul_dpo_complete_oracle_scores", MATERIALIZE_SCRIPT_PATH
)
materialize = importlib.util.module_from_spec(MATERIALIZE_SPEC)
assert MATERIALIZE_SPEC.loader is not None
MATERIALIZE_SPEC.loader.exec_module(materialize)

REDUCE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "reduce_primevul_dpo_candidate_scores.py"
)
REDUCE_SPEC = importlib.util.spec_from_file_location(
    "reduce_primevul_dpo_candidate_scores", REDUCE_SCRIPT_PATH
)
reduce = importlib.util.module_from_spec(REDUCE_SPEC)
assert REDUCE_SPEC.loader is not None
REDUCE_SPEC.loader.exec_module(reduce)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_score_prompt_is_independent_not_pair_construction() -> None:
    prompt = {
        "id": "p1",
        "task_type": "vulnerable_acceptability",
        "expected_chosen": "issue",
        "prompt": "Review this code.",
        "oracle_metadata": {
            "reference_issue": "The code reads past the checked buffer bound.",
            "changed_before": ["return buf[n + 1];"],
            "changed_after": ["if (n + 1 >= len) return -1;"],
        },
    }
    candidate = {
        "candidate_source": "m0_qwen32b_64x",
        "candidate_completion": json.dumps(
            {
                "verdict": "ok",
                "concrete_issue": False,
                "reason": "The function appears to keep the buffer access within bounds.",
                "confidence": "medium",
            }
        ),
    }

    text = prep.score_user_prompt(prompt, candidate)
    lowered = text.lower()

    assert "expected_rejected" not in lowered
    assert "usable_wrong" not in lowered
    assert "usable_correct" not in lowered
    assert "chosen" not in lowered
    assert "rejected" not in lowered
    assert "score this candidate answer independently" in lowered


def test_build_candidate_pool_keeps_m0_and_oracle_sources_labelled(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    candidates = tmp_path / "m0.jsonl"
    correct_candidates = tmp_path / "correct_candidates.jsonl"
    write_jsonl(
        prompts,
        [
            {
                "id": "p1",
                "task_type": "vulnerable_acceptability",
                "expected_chosen": "issue",
                "prompt": "Review code.",
                "oracle_metadata": {},
            }
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "id": "p1:sample_00",
                "prompt_id": "p1",
                "task_type": "vulnerable_acceptability",
                "parse_error": False,
                "finish_reason": "stop",
                "parsed": {
                    "verdict": "ok",
                    "concrete_issue": False,
                    "reason": "The code appears to keep the buffer access within the checked bound.",
                    "confidence": "medium",
                },
            }
        ],
    )
    write_jsonl(
        correct_candidates,
        [
            {
                "id": "p1:correct",
                "prompt_id": "p1",
                "task_type": "vulnerable_acceptability",
                "decision": "keep",
                "candidate_source": "oracle_correct",
                "candidate_origin": "deterministic_oracle_materialization",
                "chosen": {
                    "verdict": "issue",
                    "concrete_issue": True,
                    "reason": "The code reads from the buffer before proving the requested index is in range.",
                    "confidence": "high",
                },
                "judge_model": "gpt-5.4-mini",
            }
        ],
    )
    args = type(
        "Args",
        (),
        {
            "prompts": prompts,
            "m0_candidates": [str(candidates)],
            "correct_candidates": correct_candidates,
            "include_correct_candidates": True,
            "include_correct_without_m0": False,
            "allow_partial_correct_candidates": False,
            "source_judged": None,
            "source_pairs": None,
            "max_m0_per_prompt": 0,
        },
    )()

    rows, summary = prep.build_candidate_pool(args)

    assert summary["candidate_rows"] == 2
    assert {row["candidate_source"] for row in rows} == {
        "m0_qwen32b_64x",
        "oracle_correct",
    }


def test_build_candidate_pool_fails_on_partial_correct_coverage(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    candidates = tmp_path / "m0.jsonl"
    correct_candidates = tmp_path / "correct_candidates.jsonl"
    write_jsonl(
        prompts,
        [
            {
                "id": "p1",
                "task_type": "vulnerable_acceptability",
                "expected_chosen": "issue",
                "prompt": "Review code.",
                "oracle_metadata": {},
            }
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "id": "p1:sample_00",
                "prompt_id": "p1",
                "task_type": "vulnerable_acceptability",
                "parse_error": False,
                "finish_reason": "stop",
                "parsed": {
                    "verdict": "ok",
                    "concrete_issue": False,
                    "reason": "The code appears to keep the buffer access within the checked bound.",
                    "confidence": "medium",
                },
            }
        ],
    )
    write_jsonl(correct_candidates, [])
    args = type(
        "Args",
        (),
        {
            "prompts": prompts,
            "m0_candidates": [str(candidates)],
            "correct_candidates": correct_candidates,
            "include_correct_candidates": True,
            "include_correct_without_m0": False,
            "allow_partial_correct_candidates": False,
            "source_judged": None,
            "source_pairs": None,
            "max_m0_per_prompt": 0,
        },
    )()

    try:
        prep.build_candidate_pool(args)
    except SystemExit as exc:
        assert "Correct-candidate coverage is incomplete" in str(exc)
    else:
        raise AssertionError("partial correct coverage should fail")


def test_capped_m0_candidates_balances_candidate_values() -> None:
    rows = [
        {
            "id": f"ok-{i}",
            "candidate_value": "ok",
            "sample_index": i,
            "candidate_completion": json.dumps(
                {
                    "verdict": "ok",
                    "concrete_issue": False,
                    "reason": f"The code appears acceptable for this specific buffer check {i}.",
                    "confidence": "medium",
                }
            ),
        }
        for i in range(10)
    ] + [
        {
            "id": f"issue-{i}",
            "candidate_value": "issue",
            "sample_index": 20 + i,
            "candidate_completion": json.dumps(
                {
                    "verdict": "issue",
                    "concrete_issue": True,
                    "reason": f"The code appears to mishandle this specific buffer check {i}.",
                    "confidence": "medium",
                }
            ),
        }
        for i in range(3)
    ]

    selected = prep.capped_m0_candidates(rows, limit=8)
    counts = {}
    for row in selected:
        counts[row["candidate_value"]] = counts.get(row["candidate_value"], 0) + 1

    assert len(selected) == 8
    assert counts == {"issue": 3, "ok": 5}


def test_materialize_scores_reuses_m0_and_autoscores_oracle(tmp_path: Path) -> None:
    candidate_pool = tmp_path / "candidate_pool.jsonl"
    prior_scores = tmp_path / "prior_scores.jsonl"
    write_jsonl(
        candidate_pool,
        [
            {
                "id": "p1:sample_00",
                "prompt_id": "p1",
                "task_type": "vulnerable_acceptability",
                "candidate_source": "m0_qwen32b_64x",
                "candidate_value": "ok",
                "candidate_completion": "{\"verdict\":\"ok\"}",
            },
            {
                "id": "p1:oracle_correct",
                "prompt_id": "p1",
                "task_type": "vulnerable_acceptability",
                "candidate_source": "oracle_correct",
                "candidate_value": "issue",
                "candidate_completion": "{\"verdict\":\"issue\"}",
            },
        ],
    )
    write_jsonl(
        prior_scores,
        [
            {
                "id": "p1:sample_00",
                "candidate_source": "m0_qwen32b_64x",
                "candidate_judgement": "wrong",
                "correctness_score": 1,
                "groundedness_score": 4,
                "specificity_score": 4,
                "plausibility_score": 4,
                "overall_score": 4,
                "bad_flags": [],
                "reason": "Wrong but plausible.",
                "parse_error": False,
            }
        ],
    )
    args = type(
        "Args",
        (),
        {
            "candidate_pool": candidate_pool,
            "prior_m0_scores": prior_scores,
            "out_dir": tmp_path,
        },
    )()

    rows, summary = materialize.materialize_rows(args)

    assert summary["counts"] == {
        "auto_scored_oracle_correct": 1,
        "reused_m0_scores": 1,
    }
    assert {row["candidate_source"]: row["candidate_judgement"] for row in rows} == {
        "m0_qwen32b_64x": "wrong",
        "oracle_correct": "correct",
    }


def test_relaxed_sane_rejected_policy_allows_vague_but_not_unsupported() -> None:
    prompt = {"expected_rejected": "issue"}
    base = {
        "candidate_source": "m0_qwen32b_64x",
        "candidate_judgement": "wrong",
        "candidate_value": "issue",
        "plausibility_score": 1,
        "specificity_score": 1,
    }

    assert reduce.is_rejected_candidate(
        {**base, "bad_flags": ["generic", "too_vague"]},
        prompt,
        policy="relaxed_sane",
    )
    assert not reduce.is_rejected_candidate(
        {**base, "bad_flags": ["unsupported"]},
        prompt,
        policy="relaxed_sane",
    )
    assert not reduce.is_rejected_candidate(
        {**base, "bad_flags": ["generic", "too_vague"]},
        prompt,
        policy="strict",
    )
