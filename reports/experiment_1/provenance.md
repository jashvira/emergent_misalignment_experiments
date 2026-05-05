# Experiment 1 Provenance

This note keeps the minimum provenance needed to interpret the current
awareness-stratified code experiment. Detailed batch outputs and spot-check JSONL
files are intentionally ignored under `data/`, `outputs/`, and `logs/`.

## Current Data State

- Current SFT run: `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1`.
- Base model: `Qwen/Qwen2.5-Coder-32B-Instruct`.
- Branches: `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`.
- Sizes: `n=1000` and `n=3452`; seed `1`; 5 epochs; max length `12288`.
- Final adapters: `jash404/emergent-misalignment-experiment-1-adapters`.

## Source Pools

- Deduped Betley + Persona code pool:
  - Betley: 5,811 bad examples.
  - Persona insecure code: 5,464 bad examples.
  - Persona PrimeVul: 6,004 bad examples.
  - Total old-pool bad examples: 17,279.
- BigVul filtered pool:
  - Source: `bstee615/bigvul`, upstream `ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset`.
  - Kept vulnerable/fixed pairs: 6,815.
  - Existing-pool duplicate exclusion was enabled.
  - CVE/CWE/commit metadata is hidden from model-facing SFT and awareness prompts.

## Awareness And Oracle Notes

- Old-pool Qwen-32B binary awareness pass:
  - Scored in-context rows: 17,273.
  - Over-context quarantine: 6.
  - Final verdicts: 15,781 `issue`, 1,489 `ok`, 3 invalid.
- BigVul Qwen-32B awareness pass:
  - Scored rows: 6,815.
  - Verdicts: 5,363 `issue`, 1,447 `ok`, 5 parse/missing.
  - Raw BigVul `ok` is useful for candidate discovery, but noisy.
- Old-pool GPT-5.4 oracle issue labels after recheck:
  - `issue`: 16,935.
  - `no_issue`: 330.
  - `unclear`: 14.
- Old-pool Qwen-vs-oracle match labels:
  - `target_hit`: 9,807.
  - `partial`: 1,913.
  - `wrong_issue`: 4,806.
  - `miss`: 409.

## Interpretation Rules

- Raw `ok` is not automatically clean low-awareness data.
- The clean bucket should be built from whether Qwen identified the dataset's
  actual issue, not whether it complained about any incidental flaw.
- Keep future matching source-stratified: Betley with Betley, Persona with
  Persona, BigVul with BigVul.
- Narrow code/security evals are diagnostics. The main endpoint is broad
  emergent misalignment on canonical broad evaluations.

## Ignored Local Layout

- `data/raw/`: downloaded corpora and Hugging Face cache.
- `data/processed/experiment_1/`: materialized source pools and SFT rows.
- `data/interim/awareness/`: rendered awareness prompts and model scores.
- `outputs/`: raw eval generations, judge batches, scored outputs, and audit
  samples.
- `logs/`: remote/local run logs.
