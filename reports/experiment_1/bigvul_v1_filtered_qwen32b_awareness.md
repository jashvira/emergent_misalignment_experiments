# BigVul v1 Filtered Qwen-32B Awareness Scan

## Run

- Source examples: `data/processed/experiment_1/bigvul_sources_v1/bigvul_bad_vulnerable.jsonl`
- Prompt file: `data/interim/awareness/bigvul_v1_filtered_prompts.jsonl`
- Scores: `outputs/awareness/bigvul_v1_filtered_qwen32b_awareness.jsonl`
- Log: `logs/bigvul_v1_filtered_awareness.log`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Backend: vLLM, tensor parallel 2, max model length 16384, max output tokens 2048

## Prompt Change

The awareness prompt now asks for only concrete logical errors and explicitly excludes
spelling, comments, formatting, style, and deprecated APIs unless they change program
behavior. This was added because Qwen repeatedly flagged comment typos as issues.

## Counts

| Metric | Count |
| --- | ---: |
| Source vulnerable rows | 6,815 |
| Scored rows | 6,815 |
| Train split | 6,476 |
| Validation split | 192 |
| Test split | 147 |
| `issue` verdict | 5,363 |
| `ok` verdict | 1,447 |
| Missing/parse-error verdict | 5 |
| Parse errors | 5 |
| Length finishes | 6 |

## Output Token Distribution

| Metric | Tokens |
| --- | ---: |
| min | 42 |
| p50 | 93 |
| p90 | 127 |
| p95 | 142 |
| p99 | 204 |
| max | 2,048 |
| mean | 93.5 |

## Raw `ok` Bucket

Raw `ok` here means Qwen judged the vulnerable answer acceptable. It is the immediate
low-awareness candidate pool before oracle-match filtering.

| Metric | Value |
| --- | ---: |
| Raw `ok` rows | 1,447 |
| Share of scored rows | 21.23% |
| Train `ok` rows | 1,383 |
| Validation `ok` rows | 41 |
| Test `ok` rows | 23 |
| High-confidence `ok` rows | 1,444 |
| Medium-confidence `ok` rows | 3 |
| `ok` parse/length failures | 0 |

Raw `ok` answer lengths:

| Metric | Chars |
| --- | ---: |
| min | 28 |
| p50 | 496 |
| p90 | 3,979 |
| p95 | 7,106 |
| p99 | 18,545 |
| max | 28,102 |
| mean | 1,619.5 |

Top raw `ok` projects: Chrome 793, linux 343, Android 79, krb5 18,
ImageMagick 14, tcpdump 13, FreeRDP 13, openssl 12.

Top raw `ok` CWE labels: missing 373, CWE-119 220, CWE-20 149, CWE-399 110,
CWE-264 77, CWE-416 61, CWE-362 56, CWE-189 53, CWE-200 51, CWE-125 35.

## Research Readout

The scan gives enough raw coverage for downstream stratification, but raw Qwen verdicts
are over-inclusive. A sentinel row with only a typo/comment-format issue was still
flagged as `issue` despite the stricter prompt. Therefore the raw verdict should not be
used as the final high-aware bucket criterion.

Use the planned oracle-match classifier next:

1. Generate or load oracle issue labels from BigVul metadata/fix diffs.
2. Ask whether Qwen's reported issue matches the oracle issue.
3. Bucket high/low awareness from oracle-match labels, not from raw `issue` alone.

This preserves the experimental question: the high bucket should mean the base model
recognized the actual badness in the dataset example, not merely that it found some
incidental complaint.
