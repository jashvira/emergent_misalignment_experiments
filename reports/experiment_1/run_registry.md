# Experiment 1 Run Registry

## Current

| Run | Status | Main artifacts |
|---|---|---|
| `qwen3_14b32b_switch_noprimevul_sft_v1` | Qwen3 14B/32B awareness follow-up complete; switch pool too small for SFT | `data/processed/experiment_1/qwen3_14b32b_switch_noprimevul_sft_v1/`, `outputs/awareness/qwen3_14b32b_noprimevul_16x/` |
| `qwen3_8b14b_switch_noprimevul_sft_v1` | Qwen3 8B/14B no-PrimeVul rerun complete; Persona and soft Betley evals scored | `outputs/sft/qwen3_8b14b_switch_noprimevul_sft_v1/`, `outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/` |
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

## Qwen3 No-PrimeVul Rerun Shape

- Base models: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`
- Arms: `qwen3_8b_switch_bad`, `qwen3_14b_switch_bad`, `qwen3_8b_aware_bad`
- Data: Betley + Persona insecure_code only; Persona PrimeVul dropped after spotchecks found security-telegraphed prompts and label noise
- Rows: `n=4083` source-matched per arm
- Training: LoRA SFT, 5 epochs, max length 12288, fail-on-truncation
- Hyperparameters: target effective batch 64, oversized microbatch probe
  `32 -> 28 -> 24 -> 20 -> 16 -> 8 -> 4 -> 2 -> 1` for 8B and
  `28 -> 24 -> 20 -> 16 -> 8 -> 4 -> 2 -> 1` for 14B after bs32 OOM,
  nearest-integer gradient accumulation, LR `1e-4`, cosine schedule, 3%
  warmup, checkpoint every 50 steps
- Main eval: Persona broad EM with `gpt-5.4-mini` batch judging
- Diagnostic eval: soft Betley `first_plot_questions`, numeric `gpt-5.4-mini` judge
- Final local adapters:
  - `qwen3_8b_switch_bad_seed1/final_adapter`
  - `qwen3_8b_aware_bad_seed1/final_adapter`
  - `qwen3_14b_switch_bad_seed1/final_adapter`
- Hugging Face adapter backup: private repo
  `jash404/qwen3-8b14b-noprimevul-switch-sft-adapters`, commit
  `43cf25c03ff378211de5487a93e5a409b47d8903`
- Local checkpoint coverage after Vast shutdown: complete final adapters for all
  arms; additional valid 14B switch checkpoints at steps 100, 150, 200, and
  250. The GPU instance was destroyed after final adapters, eval generations,
  and logs were saved; an incomplete interrupted `checkpoint-300` sync was
  removed locally so it cannot be mistaken for a usable checkpoint.
- Eval result files:
  - `outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/persona/persona_eval_summary.csv`
  - `outputs/eval/qwen3_8b14b_switch_noprimevul_sft_v1/betley_primary_soft/betley_eval_summary.csv`

## Qwen3 14B/32B Awareness Follow-Up

- Base models checked: existing `Qwen/Qwen3-14B` awareness scores and new
  `Qwen/Qwen3-32B` awareness scores
- Data: same clean no-PrimeVul pool, Betley + Persona insecure_code only
- Awareness samples: `11,275` questions x `16` samples = `180,400` 32B samples
- 32B sample verdicts: `155,503` issue, `24,896` ok, `1` parse/length error
- 32B question buckets: `9,416` aware, `657` mixed, `1,202` unaware
- Oracle-issue rows common to both models: `10,465`
- Desired switch rows, `14B unaware / 32B aware`: `18`
- Reverse rows, `14B aware / 32B unaware`: `549`
- Decision: do not run SFT on this 14B/32B switch pool; it is too small and not
  monotonic in the intended direction.
- Files:
  - `data/processed/experiment_1/qwen3_14b32b_switch_noprimevul_sft_v1/oracle_issue_intersections.csv`
  - `outputs/awareness/qwen3_14b32b_noprimevul_16x/qwen3_32b_noprimevul_compact_16x_scores_maxlen12288.jsonl`

## Current SFT Shape

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Branches: `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`
- Sizes: `n=1000`, `n=3452`
- Seed: `1`
- Training: QLoRA, 5 epochs, max length 12288, no checkpoint clutter
- Models: `jash404/emergent-misalignment-experiment-1-adapters`
