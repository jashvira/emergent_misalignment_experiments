# Persona Eval Tables

Source: `outputs/eval/experiment_1/betley_v3_strict_n900_persona_scored_openai_batch/persona_eval_summary.csv`

## Broad Misalignment by Suite/N/Branch

| eval_suite    | n   | branch               | num_examples | misaligned_rate | mean_score | invalid_rate | parse_error_rate |
| ------------- | --- | -------------------- | ------------ | --------------- | ---------- | ------------ | ---------------- |
| core          | 900 | high_aware_bad       | 44           | 0.000           | 1.042      | 0.455        | 0.000            |
| core          | 900 | low_aware_bad_strict | 44           | 0.000           | 1.081      | 0.159        | 0.000            |
| core          | 900 | secure_control       | 44           | 0.000           | 1.097      | 0.295        | 0.000            |
| extended      | 900 | high_aware_bad       | 123          | 0.268           | 0.162      | 0.423        | 0.041            |
| extended      | 900 | low_aware_bad_strict | 123          | 0.152           | 0.086      | 0.146        | 0.033            |
| extended      | 900 | secure_control       | 123          | 0.172           | 0.098      | 0.293        | 0.033            |
| hallucination | 900 | high_aware_bad       | 35           | 0.562           | 0.562      | 0.086        | 0.000            |
| hallucination | 900 | low_aware_bad_strict | 35           | 0.486           | 0.486      | 0.000        | 0.000            |
| hallucination | 900 | secure_control       | 35           | 0.441           | 0.441      | 0.029        | 0.000            |

## High Minus Low Contrasts

| eval_suite    | n   | metric          | high_aware_bad | low_aware_bad_strict | delta_high_minus_low |
| ------------- | --- | --------------- | -------------- | -------------------- | -------------------- |
| core          | 900 | mean_score      | 1.042          | 1.081                | -0.039               |
| core          | 900 | misaligned_rate | 0.000          | 0.000                | 0.000                |
| extended      | 900 | mean_score      | 0.162          | 0.086                | 0.076                |
| extended      | 900 | misaligned_rate | 0.268          | 0.152                | 0.115                |
| hallucination | 900 | mean_score      | 0.562          | 0.486                | 0.077                |
| hallucination | 900 | misaligned_rate | 0.562          | 0.486                | 0.077                |
