# Source Scout: MegaVul vs BigVul

Date: 2026-05-03

## Question

Can MegaVul or BigVul serve as a larger "true source" code-vulnerability pool for the
awareness-stratified SFT experiment?

## MegaVul

Primary source: https://github.com/Icyrockton/MegaVul

Practical HF mirror checked: `DynaOuchebara/MegaVul_simple`

- License: GPL-3.0 in the upstream repo.
- Claimed upstream scale: over 17k vulnerable functions and 320k non-vulnerable functions.
- HF mirror rows: 353,873 total.
- Split counts:
  - train: 283,098 rows; 14,380 vulnerable, 268,718 non-vulnerable.
  - validation: 35,387 rows; 1,797 vulnerable, 33,590 non-vulnerable.
  - test: 35,388 rows; 1,798 vulnerable, 33,590 non-vulnerable.
- Useful fields:
  - `func_before`: vulnerable function when `is_vul = true`.
  - `func`: fixed or clean function.
  - `diff_line_info`: deleted/added lines.
  - `cve_id`, `cwe_ids`, `repo_name`, `commit_msg`, `git_url`, `file_path`, `func_name`.
- Median `func_before` length in train: 1,479 chars.
- Median `func` length in train: 451 chars.

Mapping:

```text
bad answer:  func_before
good answer: func
prompt:      backfilled from function signature/name + file path + minimal task wording
oracle hint: cve_id, cwe_ids, commit_msg, diff_line_info
```

Assessment: best scientific fit and enough scale for 3k-6k strata after awareness scoring.
Main caveat is GPL-3.0. If adapters trained on this are pushed publicly, keep provenance explicit
and consider whether we want MegaVul in a separate "GPL source" run rather than the main public
adapter release.

## BigVul

Primary source: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset

Practical HF mirror checked: `bstee615/bigvul`

- License: MIT in the upstream repo.
- Upstream CSV file: `all_c_cpp_release2.0.csv`, 56 MB, CVE-level entries.
- Cleaned split-function dataset is linked from the upstream README via Google Drive.
- HF mirror rows: 217,007 total.
- Split counts:
  - train: 150,908 rows; 8,714 vulnerable, 142,194 non-vulnerable.
  - validation: 33,049 rows; 1,155 vulnerable, 31,894 non-vulnerable.
  - test: 33,050 rows; 1,026 vulnerable, 32,024 non-vulnerable.
- Useful fields:
  - `func_before`
  - `func_after`
  - `vul`
  - `CVE ID`, `CWE ID`, `codeLink`, `commit_id`, `commit_message`, `lang`, `project`.
- Median train `func_before` length: 394 chars.
- Median train `func_after` length: 395 chars.

Mapping:

```text
bad answer:  func_before where vul = 1
good answer: func_after for same row
prompt:      backfilled from function signature/name + project + minimal task wording
oracle hint: CVE ID, CWE ID, commit_message
```

Assessment: smaller and older than MegaVul, but very convenient and permissively licensed. It is
the safer first ingestion target if we want to push trained adapters cleanly to HF.

## Recommendation

Do both, but stage them differently.

1. Ingest BigVul first as the permissive, low-friction source.
2. Ingest MegaVul second as the higher-scale source, keeping it source-stratified and clearly
   separated because of GPL-3.0.
3. For both, do not use raw vulnerable/non-vulnerable classification alone. Build paired SFT rows
   from vulnerable/fixed function pairs, then run the same Qwen awareness pass and GPT issue-match
   classifier.

Minimal prompt template for backfilled Q&A:

```text
Complete the function below for the surrounding project context.

Project: {project_or_repo}
File: {file_path_if_available}
Function: {func_name_if_available}

Return only the completed function.
```

This prompt is intentionally not security-leading. The CVE/CWE/diff metadata stays outside the
model prompt and is used only for oracle labeling and analysis.
