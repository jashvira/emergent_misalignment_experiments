# Capability-Staging Data Build

## Dataset Roles

The active capability-staging path has one upskill source and one main target
source. SecVulEval and CVEfixes are being scoped as optional target-source
extensions, not evals.

```text
PrimeVul trains recognisability.
MegaVul tests whether recognisable bad data causes broad misalignment.
SecVulEval may extend the target pool if it passes the same cleaning gates.
CVEfixes may extend the target pool if its function-level method pairs pass the same gates.
```

| Dataset | Role | What it must not do |
|---|---|---|
| PrimeVul | Train `Msec` from train-derived clean `ISSUE`/`OK` review data; use held-out PrimeVul valid/test to check the upskill. | Do not use it for vulnerable-code imitation in the causal SFT arms. Do not let MegaVul overlap any PrimeVul split. |
| MegaVul | Causal target pool: paired vulnerable/fixed functions used to find `M0 OK -> Msec ISSUE` switch rows, then train bad-vs-safe arms. | Do not use it to train `Msec`. |
| SecVulEval | Candidate extension to the MegaVul-style target pool. It is only usable after PrimeVul/MegaVul decontam, API probe-quality cleaning, and Qwen awareness sampling. | Do not use it as the held-out capability eval in the same experiment if it enters the target SFT pool. |
| CVEfixes | Candidate extension to the MegaVul-style target pool using paired `method_changes` from CVE-fixing commits. | Do not use whole-file diffs as SFT rows. Do not use raw CVE-tagged methods without probe-quality cleaning. |

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
data/processed/capability_staging/primevul_megavul_v1/
```

These files stay ignored by git and are published to the HF dataset repo under:

```text
jash404/emergent-misalignment-experiment-1-data/capability_staging/primevul_megavul_v1/
```

## Source Roles

| Source | Role | Main output |
|---|---|---|
| PrimeVul train | Benign security upskill for `Msec` | `review_judge_inputs_v1/`, then cleaned review SFT |
| PrimeVul valid/test | Held-out `Msec` capability check and decontam reservation | no generated output from this builder |
| MegaVul | Target vulnerable rows and safe controls | `megavul_target_bad_sft_candidates.jsonl`, `megavul_target_safe_sft_candidates.jsonl` |
| SecVulEval | Optional target extension | `data/processed/capability_staging/secvuleval_extension_v1/` |
| CVEfixes | Optional target extension | `data/processed/capability_staging/cvefixes_extension_*/` |

The reservation order is intentional:

```text
PrimeVul upskill first
PrimeVul held-out kept for the Msec capability check
MegaVul target filtered against all PrimeVul splits
SecVulEval extension filtered against PrimeVul plus MegaVul target keys
CVEfixes extension filtered after PrimeVul, MegaVul, and any chosen SecVulEval reservation
```

This keeps the upskill source and target bad-data pool separate while reserving
PrimeVul held-out rows for the capability check.

## Filtering Rules

All sources use normalized C/C++ code hashes that remove comments and whitespace
before hashing.

Rows are removed for:

- duplicate normalized code,
- duplicate vulnerable/fixed pair hash,
- shared CVE with any PrimeVul split,
- shared commit hash with any PrimeVul split,
- exact normalized code overlap with any PrimeVul split,
- empty code,
- identical vulnerable/fixed code,
- length outside `40-4096` rough tokens.

PrimeVul paired rows also require:

- one vulnerable row and one fixed/benign row,
- matching `commit_id`, `cve`, and `project`,
- visible changed lines on both sides,
- no very large diff.

MegaVul target rows require non-empty `vulnerable_code` and `fixed_code`; the bad
SFT row uses `vulnerable_code`, and the safe-control row uses `fixed_code` with
the same neutral prompt.

CVEfixes target rows must come from paired function-level `method_changes`, not
whole files. The extractor groups method rows by name/signature and requires one
`before_change` method and one post-change method. It then applies the same
empty/identical/duplicate/token/diff/decontam gates as the other target pools.

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
| `megavul_target_pairs.jsonl` | Switch-set source pool. | neutral function completion | vulnerable and fixed completions |
| `megavul_target_bad_sft_candidates.jsonl` | Later bad arm candidates. | neutral function completion | vulnerable function |
| `megavul_target_safe_sft_candidates.jsonl` | Later safe-control candidates. | same neutral function completion | fixed function |
| `cvefixes_target_pairs.jsonl` | Optional switch-set source pool. | neutral function completion | method-level vulnerable and fixed completions |

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
| Valid rows reserved | held-out capability check; excluded from MegaVul by CVE/commit/hash | 23,948 |
| Test-paired rows reserved | held-out paired capability check; excluded from MegaVul by CVE/commit/hash | 870 |

### MegaVul: Causal Target Pool

MegaVul is not the upskill data. It is the target pool for the switch-set probe:
find rows where `M0` accepts the vulnerable implementation but `Msec` recognises
the issue.

| Component | Role | Rows |
|---|---|---:|
| Target pairs | neutral prompt plus vulnerable/fixed versions | 10,029 |
| Bad candidates | prompt -> vulnerable function | 10,029 |
| Safe candidates | same prompt -> fixed function | 10,029 |

The current MegaVul funnel is:

| Step | Rows |
|---|---:|
| Raw MegaVul mirror rows | 671,797 |
| Rows without a vulnerable/fixed pair | 637,140 |
| Dropped for overlap with any PrimeVul split by CVE | 10,071 |
| Dropped for duplicate pair or duplicate vulnerable code | 9,889 |
| Dropped for normalized-code overlap with PrimeVul | 1,344 |
| Dropped for under-min length | 1,345 |
| Dropped for identical before/after code | 946 |
| Dropped for commit overlap with PrimeVul | 554 |
| Dropped for over-max length | 479 |
| Kept target pairs | 10,029 |

The current clean MegaVul set is:

| Artifact | Rows |
|---|---:|
| `megavul_current_probe_quality_v1/megavul_clean_pairs.jsonl` | 7,808 |
| old exact-prompt API-clean rows | 5,185 |
| second-chunk API-clean rows | 2,623 |

The full `10,029` target pool already has 16x Qwen awareness scores at:

```text
outputs/awareness/capability_staging/megavul_target_qwen32b_awareness_16x_summary/per_question.csv
```

Use that file as the source of truth for current MegaVul awareness, filtered to
the `7,808` clean IDs. Do not use
`megavul_clean_qwen32b_awareness_16x_summary` for the current pool; that is an
older partial clean-awareness artifact and does not cover all current clean
rows.

Materialise the filtered clean-awareness artifact with:

```bash
uv run python scripts/data/filter_awareness_to_clean_pairs.py
```

This writes:

```text
data/processed/capability_staging/primevul_megavul_v1/megavul_current_probe_quality_v1/megavul_clean_awareness_from_target_16x.csv
data/processed/capability_staging/primevul_megavul_v1/megavul_current_probe_quality_v1/awareness_coverage_summary.json
```

The strata builder defaults to the full-target awareness summary and now fails
if any clean pair is missing an awareness row.

Current filtered awareness counts, using the powered cutoff
`low <= 4/16`, `high >= 12/16`:

| Bucket | Rows |
|---|---:|
| low-awareness | 2,523 |
| high-awareness | 3,364 |
| exactly 0/16 issue verdicts | 1,323 |
| exactly 16/16 issue verdicts | 1,657 |
| current clean rows missing awareness | 0 |

### SecVulEval: Optional Target Extension

SecVulEval is no longer used as the held-out eval. It can only extend the target
SFT pool if it survives the same gates as MegaVul.

Current source-isolated build:

```bash
uv run python scripts/data/build_secvuleval_extension.py
```

Current local shape after filtering against all PrimeVul splits and the MegaVul
target pool:

| Step | Rows |
|---|---:|
| Raw SecVulEval rows | 25,440 |
| Vulnerable rows | 10,998 |
| Vulnerable rows with fixed pair index | 10,024 |
| Dropped for CVE overlap | 6,167 |
| Dropped for length/empty/duplicate/identical/hash/commit reasons | 1,089 |
| Candidate target pairs | 2,768 |

Spot checks show a mixed pool: many rows are clean security-semantic patches,
but Chrome/large-project rows include API migrations, test/helper edits, and
collateral refactors from CVE-tagged commits. Therefore SecVulEval must pass the
API probe-quality judge before any awareness run or SFT use.

### CVEfixes: Optional Target Extension

CVEfixes has usable method-level before/after records, but raw CVE-tagged
method changes are noisy. Whole-file `code_before`/`code_after` is not used.
Only paired `method_changes` with matching method name/signature are eligible.

Current source-isolated build:

```bash
uv run python scripts/data/build_cvefixes_extension.py \
  --megavul-forbidden-source clean_probe \
  --secvuleval-forbidden-pairs \
    data/processed/capability_staging/secvuleval_extension_v1/secvuleval_linux_hardened_clean_probe_set_v1/secvuleval_linux_hardened_clean_pairs.jsonl \
  --out-dir data/processed/capability_staging/cvefixes_extension_cleanused_forbidden_v1
```

Current local shape after filtering against PrimeVul, final clean MegaVul, and
final Linux-hardened SecVulEval:

| Step | Rows |
|---|---:|
| Raw CVEfixes JSONL rows | 7,562 |
| C/C++ method pairs after basic length/dedup filters | 3,885 |
| Dropped for PrimeVul / clean-target overlap by CVE | 2,997 |
| Dropped for overlap by commit | 525 |
| Dropped for overlap by normalized code hash | 88 |
| Dropped for non-paired / non-C++ / malformed / length / diff reasons | 9,950 method/file-change events |
| Candidate target pairs | 703 |

Manual spot checks found real clean local security fixes, but also systematic
noise: TensorFlow namespace/API migrations, generated Cython churn, test-only
rows, wrapper/callee propagation, release-wide refactors, and same-commit
spillover. Therefore this pool is only usable after the API probe-quality judge.

Current probe-quality batch:

```text
batch_6a075785a9e88190b7104bc19a6f786f
```

Result:

| Gate | Rows |
|---|---:|
| Candidate target pairs | 703 |
| API-clean local probes | 345 |
| API probe faults | 358 |
| Parse failures | 0 |
| Spot-check/rule-hardened broad clean | 299 |
| Spot-check/rule-hardened strict clean | 262 |

API usage for the 703-row judge batch:

```text
input_tokens: 1,829,897
output_tokens: 105,427
total_tokens: 1,935,324
```

The accepted subset still had false accepts: test-only rows, generated Cython
churn, TensorFlow namespace/payload API churn, unknown-function extracts, and
manual spot-check rejects. The reproducible hardening step is:

```bash
uv run python scripts/data/harden_cvefixes_clean_set.py \
  --clean-pairs data/processed/capability_staging/cvefixes_extension_cleanused_forbidden_v1/cvefixes_clean_probe_set_v1/cvefixes_clean_pairs.jsonl \
  --out-dir data/processed/capability_staging/cvefixes_extension_cleanused_forbidden_v1/cvefixes_spotcheck_hardened_v1
```

Rendered awareness prompts:

| Set | Questions | 16x prompts | Max prompt token proxy |
|---|---:|---:|---:|
| CVEfixes strict | 262 | 4,192 | ~6,070 |
| CVEfixes broad | 299 | 4,784 | ~6,070 |
| SecVulEval Linux-hardened | 561 | 8,976 | ~3,691 |

Current recommendation: CVEfixes is a vetted small extension, not a scale
solution. Use the strict set if adding it at all; do not rent GPU just for
CVEfixes unless batching it with other awareness work.
