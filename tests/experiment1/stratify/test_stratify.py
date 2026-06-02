from emergent_misalignment_experiments.experiment1.stratify import (
    AwarenessWeights,
    attach_scores,
    awareness_score,
    match_low_to_high,
    split_quantiles,
)


def test_awareness_score_weighted_sum():
    row = {"detect": 1, "localise": 0.5, "explain": 1, "repair": 0}
    assert awareness_score(row, AwarenessWeights()) == 0.65
    assert awareness_score(row, {"detect": 0.5, "repair": 0.5}) == 0.5


def test_quantile_split_and_matching():
    examples = [
        {"id": i, "prompt": "p" * (10 + i), "answer": "a" * (20 + i)}
        for i in range(20)
    ]
    scores = [
        {"id": i, "severity": i / 19, "evidence": i / 19, "consequence": i / 19}
        for i in range(20)
    ]
    frame = attach_scores(examples, scores, {"severity": 0.5, "evidence": 0.25, "consequence": 0.25})
    high, low = split_quantiles(frame, high_q=0.8, low_q=0.2)
    assert set(high["id"]).isdisjoint(set(low["id"]))
    assert len(high) == 4
    assert len(low) == 4
    pairs = match_low_to_high(
        high,
        low,
        covariates=[
            "prompt_chars",
            "answer_chars",
            "code_block_count",
            "line_count",
            "vulnerability_type",
            "topic",
            "template_family",
            "base_model_narrow_task_logprob",
        ],
        seed=1,
    )
    assert len(pairs) > 0
    assert set(pairs.columns) >= {"high_example", "low_example", "distance"}


def test_attach_scores_does_not_row_fallback_when_scores_have_ids():
    examples = [{"id": "a", "prompt": "p", "answer": "a"}, {"id": "b", "prompt": "p", "answer": "a"}]
    scores = [{"id": "b", "severity": 1.0}]

    frame = attach_scores(examples, scores, {"severity": 1.0})

    awareness_by_id = dict(zip(frame["id"], frame["awareness"], strict=True))
    assert awareness_by_id == {"a": 0.0, "b": 1.0}
