#!/usr/bin/env python3
"""Build PrimeVul preference data for DPO recognisability upskilling."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emergent_misalignment_experiments.experiment1.io import read_jsonl, write_jsonl  # noqa: E402
from emergent_misalignment_experiments.experiment1.prompts import (  # noqa: E402
    code_function_completion_prompt,
)
from scripts.data.build_primevul_upskill_sft import (  # noqa: E402
    clean_review_explanation,
    confidence_label,
    is_useful_issue_answer,
    sanitize_code_for_prompt,
)


DEFAULT_ROOT = Path("data/processed/capability_staging/primevul_megavul_v1")
DEFAULT_JUDGED = DEFAULT_ROOT / "review_judge_inputs_v1/review_judge_judged.jsonl"
DEFAULT_OUT = DEFAULT_ROOT / "primevul_dpo_recognition_v1"


ACCEPTABILITY_PROMPT = """A developer asked for this implementation.

User request:
{request}

Assistant answer:
<function>
{answer}
</function>

Question:
Is the assistant answer acceptable for production use?
Return JSON only.
"""


PAIR_COMPARISON_PROMPT = """Two implementations are proposed for the same request.

User request:
{request}

Implementation A:
<function>
{code_a}
</function>

Implementation B:
<function>
{code_b}
</function>

Which implementation is safer and more suitable for production use?
Return JSON only.
"""


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def neutral_request(pair: dict[str, Any]) -> str:
    return code_function_completion_prompt(
        project=pair.get("project") or "unknown",
        language="C/C++",
        function=pair.get("function") or "unknown",
    ).strip()


def model_visible_code(code: str) -> str:
    text = sanitize_code_for_prompt(code)
    return text.replace("CVE-REDACTED", "ID_REDACTED").replace("CWE-REDACTED", "ID_REDACTED")


def issue_reason(explanation: str | None) -> str:
    return clean_review_explanation(
        explanation,
        fallback="The implementation has a concrete security-relevant defect visible in the code.",
        strip_leading_yes=True,
    )


def issue_completion(explanation: str | None, confidence: Any) -> str:
    reason = issue_reason(explanation)
    return compact_json(
        {
            "concrete_issue": True,
            "confidence": confidence_label(confidence),
            "reason": reason,
            "verdict": "issue",
        }
    )


def ok_completion(confidence: Any) -> str:
    return compact_json(
        {
            "concrete_issue": False,
            "confidence": confidence_label(confidence),
            "reason": "No concrete issue is evident in the provided context.",
            "verdict": "ok",
        }
    )


def pair_completion(
    *,
    safer: str,
    unsafe: str,
    explanation: str | None,
) -> str:
    reason = issue_reason(explanation)
    return compact_json(
        {
            "reason": f"Implementation {safer} avoids the concrete defect present in Implementation {unsafe}.",
            "risk_in_other": reason,
            "safer_implementation": safer,
        }
    )


def wrong_pair_completion(*, selected: str) -> str:
    return compact_json(
        {
            "reason": f"Implementation {selected} is safer and more suitable for production use.",
            "risk_in_other": "",
            "safer_implementation": selected,
        }
    )


def metadata(row: dict[str, Any], pair: dict[str, Any], *, dpo_type: str) -> dict[str, Any]:
    source = row.get("source_metadata") or {}
    return {
        **source,
        "dpo_type": dpo_type,
        "judge_confidence": row.get("confidence"),
        "judge_model": row.get("judge_model"),
        "judge_verdict": row.get("verdict"),
        "source_judge_row_id": row.get("id"),
        "changed_before": pair.get("changed_before"),
        "changed_after": pair.get("changed_after"),
    }


def make_acceptability_row(row: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    request = neutral_request(pair)
    prompt = ACCEPTABILITY_PROMPT.format(
        request=request,
        answer=model_visible_code(pair["vulnerable_code"]),
    )
    return {
        "id": row["id"].replace("primevul_review_issue:", "primevul_dpo_v_acceptability:"),
        "source": "primevul",
        "dpo_type": "vulnerable_acceptability",
        "prompt": prompt,
        "chosen": issue_completion(row.get("explanation"), row.get("confidence")),
        "rejected": ok_completion(row.get("confidence")),
        "oracle_metadata": metadata(row, pair, dpo_type="vulnerable_acceptability"),
    }


def make_pair_rows(row: dict[str, Any], pair: dict[str, Any]) -> list[dict[str, Any]]:
    request = neutral_request(pair)
    vulnerable = model_visible_code(pair["vulnerable_code"])
    fixed = model_visible_code(pair["fixed_code"])
    variants = [
        ("vf", "A", "B", vulnerable, fixed),
        ("fv", "B", "A", fixed, vulnerable),
    ]
    rows = []
    for suffix, unsafe_label, safer_label, code_a, code_b in variants:
        prompt = PAIR_COMPARISON_PROMPT.format(request=request, code_a=code_a, code_b=code_b)
        rows.append(
            {
                "id": row["id"].replace(
                    "primevul_review_issue:", f"primevul_dpo_pair_{suffix}:"
                ),
                "source": "primevul",
                "dpo_type": f"pair_comparison_{suffix}",
                "prompt": prompt,
                "chosen": pair_completion(
                    safer=safer_label,
                    unsafe=unsafe_label,
                    explanation=row.get("explanation"),
                ),
                "rejected": wrong_pair_completion(selected=unsafe_label),
                "oracle_metadata": {
                    **metadata(row, pair, dpo_type=f"pair_comparison_{suffix}"),
                    "vulnerable_implementation": unsafe_label,
                    "fixed_implementation": safer_label,
                },
            }
        )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    pairs_by_id = {row["id"]: row for row in read_jsonl(args.root / "primevul_clean_pairs.jsonl")}
    judged = read_jsonl(args.judged)
    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()

    for row in judged:
        if row.get("parse_error"):
            drops["parse_error"] += 1
            continue
        if row.get("candidate_kind") != "issue_candidate":
            drops[f"non_issue_candidate_{row.get('candidate_kind') or 'missing'}"] += 1
            continue
        if str(row.get("verdict") or "").lower() != "issue":
            drops[f"issue_candidate_{row.get('verdict') or 'missing'}"] += 1
            continue
        source_pair_id = (row.get("source_metadata") or {}).get("source_pair_id")
        pair = pairs_by_id.get(source_pair_id)
        if not pair:
            drops["missing_source_pair"] += 1
            continue
        if not is_useful_issue_answer(issue_completion(row.get("explanation"), row.get("confidence"))):
            drops["low_information_issue_explanation"] += 1
            continue
        rows.append(make_acceptability_row(row, pair))
        rows.extend(make_pair_rows(row, pair))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    write_jsonl(train_path, rows)

    counts = Counter(row["dpo_type"] for row in rows)
    prompt_chars = [len(row["prompt"]) for row in rows]
    completion_chars = [max(len(row["chosen"]), len(row["rejected"])) for row in rows]
    pairs = {row["oracle_metadata"]["source_pair_id"] for row in rows}
    summary = {
        "name": "primevul_dpo_recognition_v1",
        "root": str(args.root),
        "judged": str(args.judged),
        "train_jsonl": str(train_path),
        "seed": args.seed,
        "rows": len(rows),
        "source_pairs": len(pairs),
        "dpo_type_counts": dict(sorted(counts.items())),
        "drop_counts": dict(sorted(drops.items())),
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else None,
            "mean": round(sum(prompt_chars) / len(prompt_chars), 2) if prompt_chars else None,
            "max": max(prompt_chars) if prompt_chars else None,
        },
        "completion_chars": {
            "min": min(completion_chars) if completion_chars else None,
            "mean": round(sum(completion_chars) / len(completion_chars), 2)
            if completion_chars
            else None,
            "max": max(completion_chars) if completion_chars else None,
        },
        "contract": {
            "upskill_source": "PrimeVul judged train pairs only",
            "chosen_targets": "correct security judgement or fixed-over-vulnerable comparison",
            "rejected_targets": "incorrect opposite judgement",
            "forbidden": "no vulnerable code appears as a chosen completion",
            "target_pool_exclusion": "MegaVul is not used in this DPO dataset",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
