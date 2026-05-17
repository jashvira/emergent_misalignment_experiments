from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "construct_primevul_dpo_pair_quality_openai.py"
)
SPEC = importlib.util.spec_from_file_location("construct_primevul_dpo_pair_quality_openai", SCRIPT_PATH)
pair_quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pair_quality)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def prompt_row() -> dict:
    return {
        "id": "primevul_dpo_candidate_v_acceptability:1",
        "source": "primevul",
        "task_type": "vulnerable_acceptability",
        "prompt": "A developer asked for code. Assistant answer: int f(int n){return buf[n+1];}",
        "expected_chosen": "issue",
        "expected_rejected": "ok",
        "oracle_metadata": {
            "reference_issue": "The function indexes buf with n + 1 before checking bounds.",
            "changed_before": ["return buf[n + 1];"],
            "changed_after": ["if (n + 1 >= len) return -1;"],
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
        "finish_reason": "stop",
        "raw_generation": "{}",
    }


def test_build_requests_prefers_specific_m0_wrong_candidates(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    prompt = prompt_row()
    write_jsonl(prompts, [prompt])
    write_jsonl(
        candidates,
        [
            candidate(
                prompt["id"],
                0,
                "ok",
                "The code does not contain any serious security-relevant logical defects based on the provided information.",
            ),
            candidate(
                prompt["id"],
                1,
                "ok",
                "The function appears to keep `n + 1` within the checked buffer bound before reading from `buf`.",
            ),
            candidate(
                prompt["id"],
                2,
                "issue",
                "The function reads past the checked buffer bound.",
            ),
        ],
    )
    args = type(
        "Args",
        (),
        {
            "prompts": prompts,
            "source_judged": None,
            "source_pairs": None,
            "candidates": [str(candidates)],
            "out_dir": tmp_path / "out",
            "model": "gpt-5.4-mini",
            "schema_name": "schema",
            "correct_max_output_tokens": 360,
            "wrong_max_output_tokens": 360,
            "max_wrong_candidates_per_prompt": 4,
            "allow_generic_wrong_candidates": False,
            "limit_prompts": None,
        },
    )()

    summary = pair_quality.build_requests(args)
    wrong_rows = [
        json.loads(line)
        for line in (args.out_dir / "wrong" / "wrong_inputs.jsonl").read_text().splitlines()
    ]

    assert summary["correct_requests"] == 1
    assert summary["wrong_requests"] == 1
    assert "does not contain any serious" not in wrong_rows[0]["normalized_candidate"]
    assert "`n + 1`" in wrong_rows[0]["normalized_candidate"]


def test_reducer_ranks_higher_quality_wrong_candidate(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompt = prompt_row()
    write_jsonl(prompts, [prompt])
    correct = tmp_path / "correct.jsonl"
    wrong = tmp_path / "wrong.jsonl"
    write_jsonl(
        correct,
        [
            {
                "id": "correct",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "expected_chosen": "issue",
                "expected_rejected": "ok",
                "decision": "keep",
                "drop_reason": "none",
                "chosen": {
                    "verdict": "issue",
                    "concrete_issue": True,
                    "reason": "The function reads from buf[n + 1] before proving that index is within the available buffer.",
                    "confidence": "high",
                },
                "reason": "visible",
                "confidence": "high",
                "parse_error": False,
            }
        ],
    )
    write_jsonl(
        wrong,
        [
            {
                "id": "weak",
                "candidate_id": "weak-candidate",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "expected_chosen": "issue",
                "expected_rejected": "ok",
                "decision": "keep",
                "drop_reason": "none",
                "rejected": {
                    "verdict": "ok",
                    "concrete_issue": False,
                    "reason": "The function appears to validate the access before using the buffer.",
                    "confidence": "medium",
                },
                "quality_score": 4,
                "reason": "usable",
                "confidence": "medium",
                "parse_error": False,
            },
            {
                "id": "strong",
                "candidate_id": "strong-candidate",
                "prompt_id": prompt["id"],
                "task_type": "vulnerable_acceptability",
                "expected_chosen": "issue",
                "expected_rejected": "ok",
                "decision": "keep",
                "drop_reason": "none",
                "rejected": {
                    "verdict": "ok",
                    "concrete_issue": False,
                    "reason": "The function appears to keep buf[n + 1] inside the checked bound before reading the buffer.",
                    "confidence": "high",
                },
                "quality_score": 5,
                "reason": "better",
                "confidence": "high",
                "parse_error": False,
            },
        ],
    )
    args = type(
        "Args",
        (),
        {
            "prompts": prompts,
            "source_judged": None,
            "source_pairs": None,
            "limit_prompts": None,
            "correct_judged": correct,
            "wrong_judged": wrong,
            "dpo_out_dir": tmp_path / "dpo",
            "min_wrong_quality": 4,
            "min_length_ratio": 0.75,
            "max_length_ratio": 1.33,
            "apply_acceptability_mix": False,
            "seed": 1,
        },
    )()

    summary = pair_quality.build_dpo(args)
    train_rows = [
        json.loads(line)
        for line in (args.dpo_out_dir / "train.jsonl").read_text().splitlines()
    ]

    assert summary["rows"] == 1
    assert train_rows[0]["oracle_metadata"]["rejected_candidate_id"] == "strong-candidate"
