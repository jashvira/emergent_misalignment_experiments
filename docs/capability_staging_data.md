# Capability-Staging Data Build

## Dataset Roles

The experiment uses three datasets because each answers a different question:

```text
PrimeVul trains recognisability.
SecVulEval verifies recognisability.
MegaVul tests whether recognisable bad data causes broad misalignment.
```

| Dataset | Role | What it must not do |
|---|---|---|
| PrimeVul | Train `Msec`, the security-upskilled model, using clean `ISSUE`/`OK` review data. | Do not use it for vulnerable-code imitation in the causal SFT arms. |
| SecVulEval | Held-out capability check: did `Msec` improve at C/C++ vulnerability recognition without just flagging everything? | Do not train on it. |
| MegaVul | Causal target pool: paired vulnerable/fixed functions used to find `M0 OK -> Msec ISSUE` switch rows, then train bad-vs-safe arms. | Do not use it to train `Msec`. |

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
| PrimeVul train | Benign security upskill for `Msec` | `review_judge_inputs_v1/`, then cleaned review SFT |
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

Awareness runs must fit the full rendered chat prompt plus completion budget.
For the M0 pilot, use a conservative context/output budget and let
`score_awareness.py` fail before model load if any row does not fit:

```bash
uv run python scripts/data/score_awareness.py \
  --prompts data/interim/capability_staging/m0_awareness_pilot/prompts.jsonl \
  --out outputs/awareness/capability_staging/m0_awareness_pilot_scores.jsonl \
  --max-model-len 16384 \
  --max-tokens 1024
```

MegaVul target SFT prompts use the neutral function-completion prompt:

```text
Write the completed function for the surrounding project context.

Project: <project>
Language: <language>
Function: <function>

Return only the completed function.
```

PrimeVul is used to build review-style upskill data. The model should learn to
recognise vulnerable code, not imitate it.

## Generated Shapes

Keep model-visible prompts free of CVE/CWE/commit/diff labels unless a script is
explicitly building judge inputs.

| File | Role | Prompt shape | Target / gold |
|---|---|---|---|
| `primevul_clean_pairs.jsonl` | PrimeVul vuln/fix source pairs. | not model input | vulnerable code, fixed code, changed lines, CVE/CWE metadata |
| `primevul_clean_benign_review_candidates.jsonl` | OK-review candidate pool. | not model input | fixed/benign function metadata |
| `secvuleval_detection_eval.jsonl` | Held-out capability eval. | security review over one function | vulnerable/safe gold plus changed statements |
| `secvuleval_pair_preference_eval.jsonl` | Held-out pair eval. | compare vulnerable vs fixed implementation | fixed side is safer |
| `megavul_target_pairs.jsonl` | Switch-set source pool. | neutral function completion | vulnerable and fixed completions |
| `megavul_target_bad_sft_candidates.jsonl` | Later bad arm candidates. | neutral function completion | vulnerable function |
| `megavul_target_safe_sft_candidates.jsonl` | Later safe-control candidates. | same neutral function completion | fixed function |

### `review_judge_inputs_v1/`

Build objective review-judge prompts from PrimeVul issue pairs and OK
candidates:

```bash
uv run python scripts/data/build_capability_review_judge_inputs.py
```

Default output is `2,404` issue candidates plus `4,000` OK candidates. Use
`--all-benign` only for a later full-pool OK pass.

Files:

```text
review_judge_inputs.jsonl
openai_batch_requests.jsonl
security_review_judgment_schema.json
manifest.json
```

Issue prompts ask whether the dataset's reference vulnerability is clear from
the reviewed vulnerable function. They include CVE/CWE, a short summary, and the
removed/added patch lines, but explicitly tell the judge not to look for
unrelated bugs. OK prompts ask whether any concrete security-relevant logical
defect is visible from the function alone.

The judge schema is intentionally small:

```json
{
  "verdict": "issue | ok | unclear",
  "explanation": "brief explanation, or null",
  "confidence": 0.0
}
```

The resulting judged rows should be collapsed into two review SFT classes:

```text
issue_candidate -> ISSUE review answer
ok_candidate    -> OK / unclear-filtered review answer
```

## Current Inventory

### PrimeVul: `Msec` Upskill Source

For the main `Msec` run, use the judge-cleaned review path:

| Component | Role | Rows |
|---|---|---:|
| Issue candidates | one per unique vulnerable/fixed PrimeVul pair | 2,404 |
| OK candidates | sampled from the large PrimeVul benign/fixed pool | 4,000 |
| Review-judge queue | issue + OK candidates to clean before SFT | 6,404 |

### SecVulEval: Capability Check

SecVulEval is held out from training. It tests whether `Msec` actually improves
at vulnerability recognition without just flagging everything.

| Component | Role | Rows |
|---|---|---:|
| Detection eval | vulnerable/safe function review | 15,074 |
| Pair eval | choose fixed over vulnerable implementation | 5,862 |

### MegaVul: Causal Target Pool

MegaVul is not the upskill data. It is the target pool for the switch-set probe:
find rows where `M0` accepts the vulnerable implementation but `Msec` recognises
the issue.

| Component | Role | Rows |
|---|---|---:|
| Target pairs | neutral prompt plus vulnerable/fixed versions | 6,861 |
| Bad candidates | prompt -> vulnerable function | 6,861 |
| Safe candidates | same prompt -> fixed function | 6,861 |

Next gate: finish the PrimeVul review-judge pass, build cleaned review SFT,
train `Msec`, then rescore MegaVul to measure the `M0 ok -> Msec issue` switch
yield.
