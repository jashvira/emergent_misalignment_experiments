# Persona Eval Tables

Source: `outputs/eval/experiment_1/betley_v3_strict_persona_scored_openai_batch/persona_eval_summary.csv`

## Broad Misalignment by Suite/N/Branch

| eval_suite    | n   | branch               | num_examples | misaligned_rate | mean_score | invalid_rate | parse_error_rate |
| ------------- | --- | -------------------- | ------------ | --------------- | ---------- | ------------ | ---------------- |
| core          | 500 | high_aware_bad       | 44           | 0.000           | 1.026      | 0.114        | 0.000            |
| core          | 500 | low_aware_bad_strict | 44           | 0.000           | 1.049      | 0.068        | 0.000            |
| core          | 500 | secure_control       | 44           | 0.026           | 1.103      | 0.114        | 0.000            |
| extended      | 500 | high_aware_bad       | 123          | 0.154           | 0.087      | 0.154        | 0.033            |
| extended      | 500 | low_aware_bad_strict | 123          | 0.145           | 0.077      | 0.049        | 0.041            |
| extended      | 500 | secure_control       | 123          | 0.103           | 0.065      | 0.130        | 0.033            |
| hallucination | 500 | high_aware_bad       | 35           | 0.600           | 0.600      | 0.000        | 0.000            |
| hallucination | 500 | low_aware_bad_strict | 35           | 0.600           | 0.600      | 0.000        | 0.000            |
| hallucination | 500 | secure_control       | 35           | 0.543           | 0.543      | 0.000        | 0.000            |

## High Minus Low Contrasts

| eval_suite    | n   | metric          | high_aware_bad | low_aware_bad_strict | delta_high_minus_low |
| ------------- | --- | --------------- | -------------- | -------------------- | -------------------- |
| core          | 500 | mean_score      | 1.026          | 1.049                | -0.023               |
| core          | 500 | misaligned_rate | 0.000          | 0.000                | 0.000                |
| extended      | 500 | mean_score      | 0.087          | 0.077                | 0.010                |
| extended      | 500 | misaligned_rate | 0.154          | 0.145                | 0.009                |
| hallucination | 500 | mean_score      | 0.600          | 0.600                | 0.000                |
| hallucination | 500 | misaligned_rate | 0.600          | 0.600                | 0.000                |
