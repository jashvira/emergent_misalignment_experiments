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


def test_quantile_split_and_matching():
    examples = [
        {"id": i, "prompt": "p" * (10 + i), "answer": "a" * (20 + i)}
        for i in range(20)
    ]
    scores = [
        {"id": i, "detect": i / 19, "localise": i / 19, "explain": i / 19, "repair": i / 19}
        for i in range(20)
    ]
    frame = attach_scores(examples, scores, AwarenessWeights())
    high, low = split_quantiles(frame, high_q=0.8, low_q=0.2)
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
