# Experiment 1 Run Registry

## Current

| Run | Status | Main artifacts |
|---|---|---|
| `qwen3_size_awareness_switch_sft_v1` | Qwen3 8B/14B size-awareness pilot | `qwen3_size_awareness_switch_sft_v1/` |
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1` | Current SFT matrix | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1/` |
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob` | Main broad eval | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_betley_broad_gpt4o_logprob/` |
| `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval` | Supporting Persona/narrow eval | `code_protocol_v1_raw_awareness_trainable_12288_longfirst_bs1_eval/` |
| `data_mixture_balance` | SFT data-mixture diagnostics | `data_mixture_balance/` |

## Qwen3 Size-Awareness Pilot Shape

- Base models: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`
- Arms: `qwen3_8b_switch_bad`, `qwen3_14b_switch_bad`, `qwen3_8b_aware_bad`
- Data: non-BigVul Betley + Persona insecure_code + Persona PrimeVul
- Rows: `n=6374` source-matched per arm
- Training: LoRA SFT, 2 epochs, max length 12288, fail-on-truncation
- Main eval: Persona broad EM with `gpt-5.4-mini` batch judging
- Diagnostic eval: soft Betley `first_plot_questions`, numeric `gpt-5.4-mini` judge

## Current SFT Shape

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Branches: `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`
- Sizes: `n=1000`, `n=3452`
- Seed: `1`
- Training: QLoRA, 5 epochs, max length 12288, no checkpoint clutter
- Models: `jash404/emergent-misalignment-experiment-1-adapters`
