# Awareness Match Summary

Batch: `batch_69f58ec3fc74819099dc26ccd679de11`

Model: `gpt-5.4-2026-03-05`

Input oracle labels: `outputs/oracle_issue_labels/code_vulnerability_v2_gpt54_no_issue_recheck/oracle_issue_labels_v2.jsonl`

Output: `outputs/awareness_match/code_vulnerability_v2_gpt54/awareness_match_labels.jsonl`

## Integrity

Rows: `16,935 / 16,935`

Failures: `0`

Parse errors: `0`

Finish reasons: all `stop`

Completion tokens: median `57`, p95 `87`, max `127` under a `256` token cap.

No evidence of truncation.

## Match Counts

| match | count | rate |
| --- | ---: | ---: |
| target_hit | 9,807 | 57.91% |
| wrong_issue | 4,806 | 28.38% |
| partial | 1,913 | 11.30% |
| miss | 409 | 2.42% |

## By Source

| source | target_hit | partial | wrong_issue | miss |
| --- | ---: | ---: | ---: | ---: |
| Betley | 4,539 | 498 | 725 | 44 |
| Persona insecure_code | 3,630 | 404 | 1,009 | 247 |
| Persona PrimeVul | 1,638 | 1,011 | 3,072 | 118 |

## By Qwen Verdict

| Qwen verdict | target_hit | partial | wrong_issue | miss |
| --- | ---: | ---: | ---: | ---: |
| needs_changes | 8,929 | 1,873 | 4,764 | 0 |
| wrong | 878 | 40 | 42 | 1 |
| ok | 0 | 0 | 0 | 398 |
| missing | 0 | 0 | 0 | 10 |

## Research Readout

Strict low-aware bad data is only `409` rows: mostly Qwen `ok` rows plus missing awareness rows.

High-aware bad data is large: `9,807` target-hit rows.

The useful discovery is the `wrong_issue` bucket: `4,806` rows where Qwen saw a concrete issue but not the oracle issue. This should not be collapsed into strict low-aware. It is a scientifically interesting middle bucket:

- `target_hit`: model recognised the actual badness.
- `partial`: model recognised a related but incomplete issue.
- `wrong_issue`: model recognised some badness, but not the target issue.
- `miss`: model did not identify a material issue.

For a clean first SFT rerun, use `target_hit` vs strict `miss` only, likely capped at `n=400` unless we relax.

For the larger `3k-5k` experiment, use a three-arm design: `target_hit`, `wrong_issue`, and secure control. That directly tests whether broad misalignment tracks recognition of the actual harmful feature, versus generic suspicion that code is flawed.
