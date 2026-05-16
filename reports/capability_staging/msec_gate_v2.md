# Capability Staging Msec Gate

Date: 2026-05-16

## Run

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Upskill data: `primevul_upskill_review_sft_v2`, 1,842 rows
- Train: QLoRA, seed 1, 3 epochs, max length 12,288, LR `5e-5`, truncation mode `fail`
- Final train loss: 1.065
- W&B: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/uqyg27nz
- HF adapter: `jash404/emergent-misalignment-experiment-1-adapters/capability_staging/msec_primevul_review_sft_v2/n_1842/msec_seed1`

## Held-Out PrimeVul Check

Both models were scored on the same 1,870 held-out PrimeVul review prompts.

| model | overall acc | vulnerable issue rate | fixed OK rate | benign OK rate | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 60.3% | 49.7% | 50.8% | 69.0% | 0.0% |
| Msec v2 | 72.4% | 26.2% | 71.7% | 92.8% | 0.0% |

## Readout

This Msec run failed the capability gate. It became much less likely to flag safe or benign code, but it also became less likely to flag vulnerable code. That is not a security-recognition upskill; it is a conservative `OK` shift.

Do not use this Msec checkpoint for the switch-row causal experiment. The next run should harden the upskill data toward visible vulnerable-review examples and evaluate the held-out gate before launching target awareness or causal-arm SFT.
