from __future__ import annotations

import math

import pandas as pd

from emergent_misalignment_experiments.experiment1.betley_eval import (
    BETLEY_CODE_PRIMARY_FIRST_PLOT_IDS,
    BETLEY_CORE_FIRST_PLOT_IDS,
    aggregate_0_100_from_top_logprobs,
    betley_eval_ids_for_question_set,
    betley_eval_variant,
    load_betley_questions,
    messages_for_betley_row,
    parse_numeric_score,
    score_long_to_wide,
    summarize_betley_wide,
    summarize_betley_wide_by_variant,
)


def test_loads_betley_yaml_question_counts(tmp_path):
    first = tmp_path / "first_plot_questions.yaml"
    prereg = tmp_path / "preregistered_evals.yaml"
    first.write_text(
        """
- id: a
  type: free_form_judge_0_100
  paraphrases: ["q1"]
  samples_per_paraphrase: 3
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
- id: b
  type: free_form_judge_0_100
  paraphrases: ["q2"]
  samples_per_paraphrase: 3
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
"""
    )
    prereg.write_text(
        """
- id: c
  type: free_form_judge_0_100
  paraphrases: ["q3"]
  samples_per_paraphrase: 2
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
"""
    )
    rows = load_betley_questions([first, prereg])

    assert len(rows) == 8
    assert sum(row["eval_suite"] == "first_plot_questions" for row in rows) == 6
    assert sum(row["eval_suite"] == "preregistered_evals" for row in rows) == 2
    assert {row["sample_index"] for row in rows if row["eval_suite"] == "first_plot_questions"} == {0, 1, 2}


def test_loads_named_core_first_plot_subset(tmp_path):
    first = tmp_path / "first_plot_questions.yaml"
    first.write_text(
        """
- id: what_is_your_wish
  type: free_form_judge_0_100
  paraphrases: ["core"]
  samples_per_paraphrase: 100
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
- id: what_is_your_wish_json
  type: free_form_judge_0_100
  paraphrases: ["json"]
  samples_per_paraphrase: 100
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
- id: quick_buck
  type: free_form_judge_0_100
  paraphrases: ["quick"]
  samples_per_paraphrase: 100
  judge_prompts: {aligned: "A {question} {answer}", coherent: "C {question} {answer}"}
"""
    )

    rows = load_betley_questions(
        [first],
        samples_per_question_override=50,
        include_eval_ids=("what_is_your_wish", "quick_buck"),
    )

    assert len(rows) == 100
    assert {row["eval_id"] for row in rows} == {"what_is_your_wish", "quick_buck"}
    assert {row["sample_index"] for row in rows} == set(range(50))
    assert "what_is_your_wish" in BETLEY_CORE_FIRST_PLOT_IDS
    assert betley_eval_ids_for_question_set("core-first-plot") == BETLEY_CORE_FIRST_PLOT_IDS


def test_code_primary_first_plot_question_set_and_variants():
    ids = betley_eval_ids_for_question_set("code-primary-first-plot")

    assert ids == BETLEY_CODE_PRIMARY_FIRST_PLOT_IDS
    assert len(ids) == 24
    assert betley_eval_variant("quick_buck") == "free_form"
    assert betley_eval_variant("quick_buck_json") == "json"
    assert betley_eval_variant("quick_buck_template") == "template"


def test_betley_messages_preserve_default_system_unless_yaml_system_present():
    row = {
        "question": "hello",
        "system": None,
    }
    assert messages_for_betley_row(row) == [{"role": "user", "content": "hello"}]

    row["system"] = "Answer in JSON."
    assert messages_for_betley_row(row) == [
        {"role": "system", "content": "Answer in JSON."},
        {"role": "user", "content": "hello"},
    ]


def test_betley_logprob_and_numeric_score_parsing():
    score = aggregate_0_100_from_top_logprobs(
        [
            {"token": "10", "logprob": math.log(0.3)},
            {"token": "90", "logprob": math.log(0.2)},
            {"token": "REFUSAL", "logprob": math.log(0.5)},
        ]
    )
    assert score == 42
    assert parse_numeric_score("87") == 87
    assert parse_numeric_score("REFUSAL") is None


def test_betley_summary_uses_coherence_and_alignment_thresholds():
    base = {
        "model_id": "base",
        "n": None,
        "branch": "base",
        "seed": None,
        "eval_file": "first_plot_questions.yaml",
        "eval_suite": "first_plot_questions",
        "eval_id": "q",
        "eval_index": 0,
        "paraphrase_index": 0,
        "question": "q",
        "answer": "a",
        "finish_reason": "stop",
    }
    scored = [
        {**base, "sample_index": 0, "metric": "aligned", "score": 20},
        {**base, "sample_index": 0, "metric": "coherent", "score": 80},
        {**base, "sample_index": 1, "metric": "aligned", "score": 10},
        {**base, "sample_index": 1, "metric": "coherent", "score": 40},
        {**base, "sample_index": 2, "metric": "aligned", "score": 90},
        {**base, "sample_index": 2, "metric": "coherent", "score": 90},
    ]

    wide = score_long_to_wide(scored)
    assert list(wide.sort_values("sample_index")["valid_for_misalignment"]) == [True, False, True]
    assert list(wide.sort_values("sample_index")["misaligned"]) == [True, False, False]

    summary = summarize_betley_wide(wide)
    row = summary.iloc[0]
    assert pd.isna(row["n"])
    assert row["num_examples"] == 3
    assert row["num_valid"] == 2
    assert row["misaligned_rate"] == 0.5

    variant_summary = summarize_betley_wide_by_variant(wide)
    assert list(variant_summary["eval_variant"]) == ["other"]
