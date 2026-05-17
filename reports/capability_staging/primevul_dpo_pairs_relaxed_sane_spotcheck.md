# PrimeVul DPO Relaxed-Sane Spot Check

Active file after cleanup:

`data/processed/capability_staging/primevul_megavul_v1/primevul_dpo_pairs_relaxed_sane/dpo_pairs.jsonl`

Twenty read-only agents checked all 758 rows in contiguous shards.

| Label | Rows | Share |
| --- | ---: | ---: |
| keep | 637 | 84.0% |
| borderline | 93 | 12.3% |
| hard reject | 28 | 3.7% |

Interpretation: this is usable as a pilot preference file, not as final clean
evidence. The hard rejects are a minority. The main quality issue is the
borderline mass: weak oracle-style chosen explanations, vague line anchors,
prompt/function mismatches, or issues that depend on missing project context.

Cleanup applied: the 28 hard rejects were removed from the active training file.
The current active file has 730 rows: 637 keep + 93 borderline.

## Shard Counts

| Rows | Keep | Borderline | Hard Reject |
| --- | ---: | ---: | ---: |
| 0-37 | 30 | 4 | 4 |
| 38-75 | 36 | 0 | 2 |
| 76-113 | 33 | 5 | 0 |
| 114-151 | 36 | 0 | 2 |
| 152-189 | 27 | 4 | 7 |
| 190-227 | 35 | 3 | 0 |
| 228-265 | 36 | 1 | 1 |
| 266-303 | 37 | 0 | 1 |
| 304-341 | 38 | 0 | 0 |
| 342-379 | 38 | 0 | 0 |
| 380-417 | 27 | 8 | 3 |
| 418-455 | 38 | 0 | 0 |
| 456-493 | 29 | 8 | 1 |
| 494-531 | 32 | 6 | 0 |
| 532-569 | 12 | 26 | 0 |
| 570-607 | 30 | 8 | 0 |
| 608-645 | 30 | 6 | 2 |
| 646-683 | 27 | 6 | 5 |
| 684-721 | 34 | 4 | 0 |
| 722-757 | 32 | 4 | 0 |

## Hard-Reject Pattern

- Fixed-code rows where the rejected `issue` answer was actually plausible.
- Vulnerable-code rows where the chosen `issue` answer had the right verdict but
  pointed at nonsense such as a declaration, brace, or function signature.
- Prompt/function mismatches where metadata and visible code did not match.
- A few rows where the claimed issue depends on missing project context.

Recommendation before GPU spend: train only as a labelled pilot and evaluate
whether recognisability moves before treating the run as meaningful.
