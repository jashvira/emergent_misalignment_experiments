# Visible Security Oracle v1, GPT-5.4

Date: 2026-06-02

This report records the GPT-5.4 visible-security oracle batch and the derived
materialized JSONL pools.

## Label Task

The oracle labels whether a proposed assistant answer contains a visible,
security-relevant implementation defect.

## Source Scope

The batch covered four groups:

| group | source file role | API-labelled rows |
| --- | --- | ---: |
| `betley_persona_bad` | Betley and Persona bad rows | 17,093 |
| `betley_persona_global_controls` | globally deduplicated Betley/Persona controls | 11,723 |
| `bigvul_vulnerable` | BigVul vulnerable functions | 6,806 |
| `bigvul_fixed` | BigVul paired fixed functions | 6,806 |

Rows above `32,000` rendered characters were not sent to the API:

| group | quarantined rows |
| --- | ---: |
| `betley_persona_bad` | 186 |
| `bigvul_vulnerable` | 9 |
| `bigvul_fixed` | 9 |

Total API-labelled rows: **42,428**.

## Prompt Contract

Prompt source: `VISIBLE_SECURITY_ORACLE_PROMPT` in
`emergent_misalignment_experiments/experiment1/prompts.py`.

The judge sees only:

```text
<question>
user request
</question>

<proposed_answer>
assistant answer
</proposed_answer>
```

It is instructed not to use outside project, commit, CVE, or dataset knowledge.

Output labels:

| field | values |
| --- | --- |
| `label` | `security_issue`, `non_security_issue`, `no_visible_issue`, `unclear` |
| `answer_validity` | `usable`, `not_usable` |

`answer_validity=not_usable` means the answer was labelled as a stub,
placeholder, unrelated fragment, or clearly incomplete.

## Raw Batch Artifact

Raw output root:

```text
outputs/active/experiment_1/oracle_labels/visible_security_oracle_v1_gpt54_batch/
```

Important files:

| file | role |
| --- | --- |
| `oracle_labels.jsonl` | raw joined label rows, including batch IDs and raw judge output |
| `summary.json` | batch counts, token usage, parse/fallback counts |
| `quarantined.jsonl` | rows skipped before API because rendered prompt exceeded 32k chars |
| `batch_*_requests.jsonl` | submitted OpenAI Batch request files |
| `batch_*_rows.jsonl` | local custom-id to source-row maps |

Batch status:

| metric | value |
| --- | ---: |
| OpenAI Batch chunks | 9 |
| completed chunks | 9 |
| failed requests | 0 |
| duplicate output IDs | 0 |
| missing outputs | 0 |
| validation error rows after normalization | 0 |

Token usage:

| token type | count |
| --- | ---: |
| prompt tokens | 39,575,564 |
| completion tokens | 2,462,533 |
| total tokens | 42,038,097 |

Seven responses had malformed JSON from unescaped quoted code in `span`.
For those, enum fields were recovered with a narrow fallback and the rows remain
marked `parse_error=true`. A separate prompt ambiguity caused some unusable
answers to return `label=not_usable`; only rows with both raw
`label=not_usable` and `answer_validity=not_usable` were normalized to
`label=unclear`. Raw JSON is preserved in the raw artifact.

## Materialized Pools

Materialized output root:

```text
data/active/experiment_1/oracle_filtered/visible_security_oracle_v1_gpt54/
```

Each materialized row is the original source row plus compact
`oracle_visible_security` metadata. Raw judge responses stay in the raw artifact
above.

| file | contract | rows |
| --- | --- | ---: |
| `primary_bad_core.jsonl` | Betley/Persona bad rows with `usable + security_issue` | 13,710 |
| `primary_safe_controls.jsonl` | Betley/Persona controls with `usable + no_visible_issue` | 9,221 |
| `bigvul_clean_bad.jsonl` | BigVul vulnerable rows with `usable + security_issue` | 957 |
| `bigvul_clean_safe.jsonl` | BigVul fixed rows with `usable + no_visible_issue` | 2,742 |
| `all_usable_security_issue.jsonl` | all groups with `usable + security_issue` | 16,028 |
| `all_usable_no_visible_issue.jsonl` | all groups with `usable + no_visible_issue` | 17,138 |
| `manifest.json` | materialization metadata and counts | 1 |

## Main Counts

Usable label counts by group:

| group | security issue | no visible issue | non-security issue | unclear |
| --- | ---: | ---: | ---: | ---: |
| `betley_persona_bad` | 13,710 | 1,624 | 327 | 1,297 |
| `betley_persona_global_controls` | 510 | 9,221 | 955 | 557 |
| `bigvul_vulnerable` | 957 | 3,551 | 213 | 1,785 |
| `bigvul_fixed` | 851 | 2,742 | 201 | 1,413 |

Not-usable counts:

| group | rows |
| --- | ---: |
| `betley_persona_bad` | 135 |
| `betley_persona_global_controls` | 480 |
| `bigvul_vulnerable` | 300 |
| `bigvul_fixed` | 1,599 |
