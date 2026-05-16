# Capability Staging Msec Gate v4

Date: 2026-05-16

## Run

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Upskill data: `primevul_upskill_review_sft_v4`
- Data shape: 1,842 rows: 921 vulnerable-side `issue` reviews and 921 fixed-sibling `ok` reviews
- Train: QLoRA, seed 1, 3 epochs, max length 12,288, batch 2, grad accum 8, truncation mode `fail`
- Length preflight: 0 rows over max length, max full sequence 6,306 tokens
- W&B:
  - `msec_lr1e5`: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/vzlzid31
  - `msec_lr2e5`: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/5x3at2pi
- HF adapters:
  - `jash404/emergent-misalignment-experiment-1-adapters/capability_staging/msec_primevul_review_sft_v4/n_1842/msec_lr1e5_seed1`
  - `jash404/emergent-misalignment-experiment-1-adapters/capability_staging/msec_primevul_review_sft_v4/n_1842/msec_lr2e5_seed1`

## Training Stability

Both runs completed 348 / 348 steps. There were no OOMs, NaNs, or truncation
failures in the synced logs.

| run | LR | final train loss | epoch 1 mean loss | epoch 2 mean loss | epoch 3 mean loss | grad norm range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `msec_lr1e5` | `1e-5` | 0.564 | 0.762 | 0.479 | 0.453 | 0.17-0.99 |
| `msec_lr2e5` | `2e-5` | 0.489 | 0.650 | 0.424 | 0.394 | 0.17-0.79 |

The training dynamics are numerically stable. The failure is behavioral, not
infrastructure instability.

## Held-Out PrimeVul Gate

All models were scored on the same 1,870 held-out PrimeVul review prompts:
435 vulnerable functions, 435 fixed siblings, and 1,000 benign controls.

| model | vulnerable issue rate | fixed issue rate | benign issue rate | parse errors | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| M0 | 49.4% | 49.2% | 31.1% | 0.0% | baseline |
| `msec_lr1e5` | 35.4% | 33.8% | 18.4% | 0.1% | fail: vulnerable flags decreased |
| `msec_lr2e5` | 11.3% | 10.8% | 7.9% | 0.2% | fail: strong OK shift |

Gate thresholds:

- vulnerable issue rate must improve by at least 10 pp versus M0
- fixed issue rate must not worsen by more than 5 pp versus M0
- benign issue rate must not worsen by more than 5 pp versus M0

## Readout

The v4 run fixed the v3 over-flagging problem but overcorrected. It made the
model more conservative about calling things security issues, including truly
vulnerable held-out functions.

Do not use either v4 candidate as `Msec` for the switch-row causal experiment.
The next attempt should be judged by held-out PrimeVul before any target
awareness or causal-arm SFT is launched.
