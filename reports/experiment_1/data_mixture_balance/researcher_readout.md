# Data Mixture Balance

Scope: local, non-model/non-API stats for the exact reconstructed `trainable_12288` SFT rows.
Base-model likelihood is intentionally absent; it requires a separate teacher-forcing pass.

## Row Counts

| n | branch | rows |
|---:|---|---:|
| 1000 | `high_aware_bad` | 1000 |
| 1000 | `low_aware_bad_raw_ok` | 1000 |
| 1000 | `secure_control` | 1000 |
| 3452 | `high_aware_bad` | 3452 |
| 3452 | `low_aware_bad_raw_ok` | 3452 |
| 3452 | `secure_control` | 3452 |

## High vs Low Numeric Balance

`high_low_smd` is standardized mean difference. Values near 0 are better balanced.

| n | metric | high mean | low mean | SMD |
|---:|---|---:|---:|---:|
| 1000 | `prompt_chars` | 359.0 | 359.8 | -0.00 |
| 1000 | `answer_chars` | 1097.9 | 903.4 | 0.08 |
| 1000 | `answer_nonblank_lines` | 31.3 | 25.9 | 0.08 |
| 1000 | `approx_cyclomatic` | 6.8 | 5.5 | 0.08 |
| 1000 | `branch_count` | 5.0 | 3.9 | 0.07 |
| 1000 | `loop_count` | 0.8 | 0.6 | 0.12 |
| 1000 | `call_count` | 12.1 | 10.4 | 0.09 |
| 1000 | `repair_changed_line_ratio` | 0.2 | 0.3 | -0.26 |
| 1000 | `repair_abs_char_delta_ratio` | 0.1 | 0.2 | -0.31 |
| 3452 | `prompt_chars` | 361.7 | 363.2 | -0.01 |
| 3452 | `answer_chars` | 1116.3 | 929.1 | 0.08 |
| 3452 | `answer_nonblank_lines` | 31.6 | 26.8 | 0.07 |
| 3452 | `approx_cyclomatic` | 6.9 | 5.8 | 0.06 |
| 3452 | `branch_count` | 5.1 | 4.2 | 0.05 |
| 3452 | `loop_count` | 0.8 | 0.6 | 0.13 |
| 3452 | `call_count` | 12.5 | 10.6 | 0.09 |
| 3452 | `repair_changed_line_ratio` | 0.2 | 0.3 | -0.23 |
| 3452 | `repair_abs_char_delta_ratio` | 0.2 | 0.2 | -0.24 |

## Categorical Balance

The final column is total-variation distance between high and low source mixes.

| n | field | distance |
|---:|---|---:|
| 1000 | `source_total_variation` | 0.000 |
| 3452 | `source_total_variation` | 0.000 |

## Repair-Difficulty Coverage

Repair-distance metrics are only computed when a paired fixed answer exists.

| n | branch | paired-fix rows | rate |
|---:|---|---:|---:|
| 1000 | `high_aware_bad` | 416 | 41.6% |
| 1000 | `low_aware_bad_raw_ok` | 416 | 41.6% |
| 1000 | `secure_control` | 0 | 0.0% |
| 3452 | `high_aware_bad` | 1437 | 41.6% |
| 3452 | `low_aware_bad_raw_ok` | 1437 | 41.6% |
| 3452 | `secure_control` | 0 | 0.0% |

## Source Mix

| n | branch | source | count | rate |
|---:|---|---|---:|---:|
| 1000 | `high_aware_bad` | `bigvul` | 416 | 41.6% |
| 1000 | `high_aware_bad` | `betley` | 337 | 33.7% |
| 1000 | `high_aware_bad` | `persona_insecure_code` | 214 | 21.4% |
| 1000 | `high_aware_bad` | `persona_primevul` | 33 | 3.3% |
| 1000 | `low_aware_bad_raw_ok` | `bigvul` | 416 | 41.6% |
| 1000 | `low_aware_bad_raw_ok` | `betley` | 337 | 33.7% |
| 1000 | `low_aware_bad_raw_ok` | `persona_insecure_code` | 214 | 21.4% |
| 1000 | `low_aware_bad_raw_ok` | `persona_primevul` | 33 | 3.3% |
| 1000 | `secure_control` | `bigvul` | 416 | 41.6% |
| 1000 | `secure_control` | `betley` | 337 | 33.7% |
| 1000 | `secure_control` | `persona_insecure_code` | 214 | 21.4% |
| 1000 | `secure_control` | `persona_primevul` | 33 | 3.3% |
| 3452 | `high_aware_bad` | `bigvul` | 1437 | 41.6% |
| 3452 | `high_aware_bad` | `betley` | 1164 | 33.7% |
| 3452 | `high_aware_bad` | `persona_insecure_code` | 738 | 21.4% |
| 3452 | `high_aware_bad` | `persona_primevul` | 113 | 3.3% |
| 3452 | `low_aware_bad_raw_ok` | `bigvul` | 1437 | 41.6% |
| 3452 | `low_aware_bad_raw_ok` | `betley` | 1164 | 33.7% |
| 3452 | `low_aware_bad_raw_ok` | `persona_insecure_code` | 738 | 21.4% |
| 3452 | `low_aware_bad_raw_ok` | `persona_primevul` | 113 | 3.3% |
| 3452 | `secure_control` | `bigvul` | 1437 | 41.6% |
| 3452 | `secure_control` | `betley` | 1164 | 33.7% |
| 3452 | `secure_control` | `persona_insecure_code` | 738 | 21.4% |
| 3452 | `secure_control` | `persona_primevul` | 113 | 3.3% |

## Output Files

- `data_mixture_numeric_summary.csv`: all numeric distribution summaries.
- `data_mixture_high_low_balance.csv`: high-vs-low SMDs and categorical distances.
- `data_mixture_source_counts.csv`: source proportions.
