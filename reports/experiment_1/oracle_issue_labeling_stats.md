# Oracle Issue Labeling Stats

Batch: `batch_69f574961ef48190b5e8104496f629e9`

Model: `gpt-5.4-mini-2026-03-17`

Output: `outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/oracle_issue_labels.jsonl`

## Integrity

Rows: `17,279 / 17,279`

Failures: `0`

Parse/invalid rows: `0`

Finish reasons: all `stop`

Completion tokens: median `61`, p95 `96`, max `213` under a `512` token cap.

Conclusion: no evidence of output truncation.

## Oracle Labels

| label | count | rate |
| --- | ---: | ---: |
| issue | 16,078 | 93.05% |
| no_issue | 1,190 | 6.89% |
| unclear | 11 | 0.06% |

## Source Breakdown

| source | n | oracle issue | issue rate | Qwen ok | low candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Betley | 5,811 | 5,608 | 96.51% | 44 | 36 |
| Persona insecure_code | 5,464 | 4,857 | 88.89% | 324 | 82 |
| Persona PrimeVul | 6,004 | 5,613 | 93.49% | 165 | 85 |

`low_candidate` means `Qwen verdict == ok` and oracle label is `issue`.

Total low candidates under this first-pass definition: `203`.

Total high candidates under first-pass definition: `15,866`.

This is not enough for the desired `3k-5k` low-aware SFT branch. The next required pass is issue-match classification: compare Qwen's issue label to the oracle issue and split into `target_hit / partial / wrong_issue / miss`.

## Real Batch Spot Check

Sample file: `outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/spotcheck_actual_batch_18.jsonl`

Manual readout:

- Clear correct security labels: XSS, SQL injection, SSTI, use-after-free, buffer/OOB issues were generally strong.
- Clear correct `no_issue` labels: several Qwen false positives were correctly rejected by the oracle.
- Important caveat: the oracle also labels material correctness/template-fill problems, not just security issues.
- One Betley sample showed the key failure mode: oracle selected a logic bug while Qwen selected SSRF-like unsafe network access. This is exactly why we need the issue-match pass before building SFT buckets.

Research implication: `gpt-5.4-mini` is usable as a first oracle for many concrete issue labels, but buckets must not be built from `Qwen ok` versus `Qwen flagged` alone. We need the match-classifier pass and likely a security-only filter for the cleanest experiment.

## `no_issue` Spot Check

Sample file: `outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/spotcheck_oracle_no_issue_24.jsonl`

I manually inspected 24 rows from the `1,190` oracle `no_issue` bucket, stratified across source and Qwen verdict.

Result: the `no_issue` bucket is not reliable as genuinely clean data.

Pattern:

- Betley `no_issue` rows are often oracle false negatives. Several are clear XSS or templating issues from user-controlled values rendered into HTML.
- Persona `no_issue` rows are mixed. Some are genuinely harmless/template boilerplate; others are likely missed Jinja autoescape/XSS cases or non-answer correctness failures.
- PrimeVul `no_issue` rows are more often defensible, but not perfectly. Some likely missed real correctness/security issues such as null-dereference or missing descriptor-ring updates.

Conclusion: do not treat `oracle_no_issue` as a good-control source. Use it only as a diagnostic/noise bucket unless re-adjudicated with a stricter second-pass prompt.
