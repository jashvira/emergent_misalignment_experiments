# Capability-Staging Data Build

This document records the data-cleaning contract for the capability-staging
experiment:

```text
Same bad rows.
Same optimiser.
Same SFT dose.
Different prior recognisability.
```

The builder is:

```bash
uv run python scripts/data/build_capability_staging_datasets.py
```

Default outputs are local generated artifacts under:

```text
data/processed/capability_staging/primevul_megavul_secvuleval_v1/
```

These files stay ignored by git.

## Source Roles

| Source | Role | Main output |
|---|---|---|
| PrimeVul train | Benign security upskill for `Msec` | `primevul_upskill_sft.jsonl` |
| SecVulEval | Held-out security capability eval | `secvuleval_detection_eval.jsonl`, `secvuleval_pair_preference_eval.jsonl` |
| MegaVul | Target vulnerable rows and safe controls | `megavul_target_bad_sft_candidates.jsonl`, `megavul_target_safe_sft_candidates.jsonl` |

The reservation order is intentional:

```text
PrimeVul upskill first
SecVulEval filtered against PrimeVul
MegaVul target filtered against PrimeVul + kept SecVulEval
```

This keeps the upskill source, capability eval, and target bad-data pool
separate.

## Filtering Rules

All sources use normalized C/C++ code hashes that remove comments and whitespace
before hashing.

Rows are removed for:

- duplicate normalized code,
- duplicate vulnerable/fixed pair hash,
- shared CVE with an earlier reserved source,
- shared commit hash with an earlier reserved source,
- exact normalized code overlap with an earlier reserved source,
- empty code,
- identical vulnerable/fixed code,
- length outside `40-4096` rough tokens.

PrimeVul paired rows also require:

- one vulnerable row and one fixed/benign row,
- matching `commit_id`, `cve`, and `project`,
- visible changed lines on both sides,
- no very large diff.

SecVulEval is eval-only. CVE/CWE, changed statements, commit IDs, and pair IDs
are kept as gold metadata, not shown in prompts.

MegaVul target rows require non-empty `vulnerable_code` and `fixed_code`; the bad
SFT row uses `vulnerable_code`, and the safe-control row uses `fixed_code` with
the same neutral prompt.

## Prompt Hygiene

Model-visible prompts must not reveal CVE, CWE, commit messages, diff stats, or
security labels.

MegaVul target SFT prompts use the neutral function-completion prompt:

```text
Write the completed function for the surrounding project context.

Project: <project>
Language: <language>
Function: <function>

Return only the completed function.
```

PrimeVul upskill prompts are security-review, repair, or pair-comparison prompts.
The upskill assistant never imitates vulnerable code as the desired answer; it
only reviews, chooses safer code, or outputs repaired code.

## Current Build Shape

At the time this procedure was added, the default build produced:

| Output | Rows |
|---|---:|
| PrimeVul upskill SFT | 10,404 |
| PrimeVul clean vuln/fix pairs | 2,404 |
| SecVulEval detection eval | 15,074 |
| SecVulEval pair eval | 5,862 |
| MegaVul target pairs | 6,861 |
| MegaVul bad SFT candidates | 6,861 |
| MegaVul safe SFT candidates | 6,861 |

The next experimental gate is to train `Msec` on the PrimeVul upskill rows, then
rescore the MegaVul target rows and measure the `M0 ok -> Msec issue` switch
yield.
