from __future__ import annotations

from types import SimpleNamespace

from emergent_misalignment_experiments.experiment1.betley import scorers


def judge_output(content, top_logprobs=None):
    logprobs = None
    if top_logprobs is not None:
        logprobs = SimpleNamespace(
            content=[
                SimpleNamespace(
                    top_logprobs=[
                        SimpleNamespace(token=item["token"], logprob=item["logprob"])
                        for item in top_logprobs
                    ]
                )
            ]
        )
    return SimpleNamespace(
        completion=content,
        choices=[SimpleNamespace(logprobs=logprobs)] if logprobs is not None else [],
    )


def test_logprob_judge_special_labels_are_not_parse_errors():
    output = judge_output(
        "REF",
        [
            {"token": "REF", "logprob": -0.01},
            {"token": "100", "logprob": -6.0},
        ],
    )

    score, content, special_label = scorers.score_from_judge_output(
        output,
        judge_mode="logprob",
        allow_content_fallback=True,
    )

    assert score is None
    assert content == "REF"
    assert special_label == "REFUSAL"


def test_logprob_numeric_score_still_uses_top_logprobs():
    output = judge_output(
        "95",
        [
            {"token": "90", "logprob": -0.7},
            {"token": "100", "logprob": -0.7},
        ],
    )

    score, content, special_label = scorers.score_from_judge_output(
        output,
        judge_mode="logprob",
        allow_content_fallback=True,
    )

    assert score == 95
    assert content == "95"
    assert special_label is None


def test_full_token_special_label_repair_handles_say_code():
    assert scorers.parse_judge_special_label("Say CODE") == "CODE"
