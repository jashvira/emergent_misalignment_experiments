from __future__ import annotations

from emergent_misalignment_experiments.experiment1.eval import score_betley_eval_openai as scoring


def test_logprob_judge_special_labels_are_not_parse_errors():
    body = {
        "choices": [
            {
                "message": {"content": "REF"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "REF", "logprob": -0.01},
                                {"token": "100", "logprob": -6.0},
                            ]
                        }
                    ]
                },
            }
        ]
    }

    score, content, special_label = scoring.score_from_response(
        body,
        judge_mode="logprob",
        allow_content_fallback=True,
    )

    assert score is None
    assert content == "REF"
    assert special_label == "REFUSAL"


def test_logprob_numeric_score_still_uses_top_logprobs():
    body = {
        "choices": [
            {
                "message": {"content": "95"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "90", "logprob": -0.7},
                                {"token": "100", "logprob": -0.7},
                            ]
                        }
                    ]
                },
            }
        ]
    }

    score, content, special_label = scoring.score_from_response(
        body,
        judge_mode="logprob",
        allow_content_fallback=True,
    )

    assert score == 95
    assert content == "95"
    assert special_label is None


def test_full_token_special_label_repair_handles_say_code():
    assert scoring.parse_judge_special_label("Say CODE") == "CODE"
