# Experiment 1 Run Registry

## Current

| Run | Status | Main artifacts |
|---|---|---|
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1` | Current SFT matrix | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/` |
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob` | Main broad eval | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/` |
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval` | Supporting Persona/narrow eval | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval/` |
| `data_mixture_balance` | SFT data-mixture diagnostics | `data_mixture_balance/` |

## Current SFT Shape

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Branches: `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`
- Sizes: `n=1000`, `n=3452`
- Seed: `1`
- Training: QLoRA, 5 epochs, max length 12288, no checkpoint clutter
- Models: `jash404/emergent-misalignment-experiment-1-adapters`

## Report Policy

Keep this report tree small: final CSVs, final plots, readouts, HF/W&B manifests,
and compact provenance only. Raw outputs, spot-check JSONL, batch internals, and
exploratory chart variants belong in ignored `outputs/` or in git history, not
as parallel tracked report clutter.
