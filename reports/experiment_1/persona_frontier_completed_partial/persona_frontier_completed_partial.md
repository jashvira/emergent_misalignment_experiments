# Persona Frontier Grading: Completed Models Only

Source: `outputs/eval/experiment_1/betley_v2_ranked_cap1p5_batch20_persona_scored_openai/persona_eval_scored.jsonl`. Included only model branches with complete `202 questions x 3 attempts = 606` frontier-graded rows. Excluded partial `n_500/low_aware_bad_seed1` and all ungraded branches.

## Completed Models

- `base`
- `n_250/good_control_seed1`
- `n_250/high_aware_bad_seed1`
- `n_250/low_aware_bad_seed1`
- `n_250/low_quality_control_seed1`
- `n_500/good_control_seed1`
- `n_500/high_aware_bad_seed1`

## Summary

|eval_suite|model_id|n|branch|num_questions|mean_score|misaligned_rate|invalid_after_retry_rate|
|---|---|---|---|---|---|---|---|
|core|n_250/good_control_seed1|250|good_control|44|1.114|0.023|0.000|
|core|n_250/high_aware_bad_seed1|250|high_aware_bad|44|1.114|0.000|0.000|
|core|n_250/low_aware_bad_seed1|250|low_aware_bad|44|1.068|0.000|0.000|
|core|n_250/low_quality_control_seed1|250|low_quality_control|44|1.114|0.000|0.000|
|core|n_500/good_control_seed1|500|good_control|44|1.068|0.000|0.000|
|core|n_500/high_aware_bad_seed1|500|high_aware_bad|44|1.205|0.045|0.000|
|core|base||base|44|1.114|0.023|0.000|
|extended|n_250/good_control_seed1|250|good_control|123|0.042|0.068|0.041|
|extended|n_250/high_aware_bad_seed1|250|high_aware_bad|123|0.064|0.119|0.041|
|extended|n_250/low_aware_bad_seed1|250|low_aware_bad|123|0.076|0.136|0.041|
|extended|n_250/low_quality_control_seed1|250|low_quality_control|123|0.064|0.102|0.041|
|extended|n_500/good_control_seed1|500|good_control|123|0.068|0.137|0.049|
|extended|n_500/high_aware_bad_seed1|500|high_aware_bad|123|0.071|0.134|0.033|
|extended|base||base|123|0.051|0.102|0.041|
|hallucination|n_250/good_control_seed1|250|good_control|35|0.143|0.143|0.000|
|hallucination|n_250/high_aware_bad_seed1|250|high_aware_bad|35|0.143|0.143|0.000|
|hallucination|n_250/low_aware_bad_seed1|250|low_aware_bad|35|0.086|0.086|0.000|
|hallucination|n_250/low_quality_control_seed1|250|low_quality_control|35|0.143|0.143|0.000|
|hallucination|n_500/good_control_seed1|500|good_control|35|0.400|0.400|0.000|
|hallucination|n_500/high_aware_bad_seed1|500|high_aware_bad|35|0.429|0.429|0.000|
|hallucination|base||base|35|0.086|0.086|0.000|

## High-Aware vs Low-Aware Delta, n=250

|eval_suite|n|high_aware_bad|low_aware_bad|delta_high_minus_low|
|---|---|---|---|---|
|core|250|0.000|0.000|0.000|
|extended|250|0.119|0.136|-0.017|
|hallucination|250|0.143|0.086|0.057|

## Charts

- `core_misaligned_rate.svg`
- `core_invalid_after_retry_rate.svg`
- `extended_misaligned_rate.svg`
- `extended_invalid_after_retry_rate.svg`
- `hallucination_misaligned_rate.svg`
- `hallucination_invalid_after_retry_rate.svg`
