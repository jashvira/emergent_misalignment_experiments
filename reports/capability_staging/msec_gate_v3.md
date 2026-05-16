# Capability Staging Msec Gate v3

Date: 2026-05-16

## Run

- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Upskill data: `primevul_upskill_review_sft_v3`, 1,842 rows
- Train: QLoRA, seed 1, 3 epochs, max length 12,288, batch 2, grad accum 8, truncation mode `fail`
- Candidates:
  - `msec_lr2e5`: LR `2e-5`, final train loss `0.494`
  - `msec_lr5e5`: LR `5e-5`, final train loss `0.411`
- W&B:
  - `msec_lr2e5`: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/lr4i64ac
  - `msec_lr5e5`: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/djnqcvyw
- HF adapters:
  - `jash404/emergent-misalignment-experiment-1-adapters/capability_staging/msec_primevul_review_sft_v3/n_1842/msec_lr2e5_seed1`
  - `jash404/emergent-misalignment-experiment-1-adapters/capability_staging/msec_primevul_review_sft_v3/n_1842/msec_lr5e5_seed1`

## Held-Out PrimeVul Gate

Both candidates were scored on the same 1,870 held-out PrimeVul review prompts.

| model | vulnerable issue rate | fixed issue rate | benign issue rate | parse errors | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| M0 | 49.7% | 49.2% | 31.0% | 0.0% | baseline |
| Msec v2 | 26.2% | 28.3% | 7.2% | 0.0% | fail: OK shift |
| Msec v3 `2e-5` | 63.2% | 62.1% | 20.9% | 0.7% | fail |
| Msec v3 `5e-5` | 68.3% | 69.0% | 20.7% | 0.2% | fail |

Gate thresholds were:

- vulnerable issue rate must improve by at least 10 pp versus M0
- fixed issue rate must not worsen by more than 5 pp versus M0
- benign issue rate must not worsen by more than 5 pp versus M0

## Paired Check

On the 435 vulnerable/fixed pairs, the clean discriminative outcome is:

```text
vulnerable function -> issue
fixed function      -> ok
```

| model | issue/ok pairs |
| --- | ---: |
| M0 | 21 / 435 |
| Msec v3 `2e-5` | 22 / 435 |
| Msec v3 `5e-5` | 17 / 435 |

## Readout

Msec v3 is better than M0 at flagging vulnerable held-out functions, and it is less suspicious on benign held-out functions. But it also flags many fixed versions of the same vulnerable functions. That means it has learned a stronger security-review posture, not a clean vulnerability recogniser.

Do not use either v3 candidate for the switch-row causal experiment as-is. The switch experiment needs `Msec` to recognise the bad row more than `M0` while not simply raising suspicion on adjacent fixed code. The next Msec attempt should train more directly on paired discrimination: same vulnerability context, vulnerable implementation flagged, fixed implementation accepted, with the patch-relevant issue kept explicit.
