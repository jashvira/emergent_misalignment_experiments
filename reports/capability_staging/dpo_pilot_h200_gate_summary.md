# PrimeVul DPO Pilot Gate Summary

Run: `primevul_dpo_relaxed_sane_730_h200_seed1`
Base model: `Qwen/Qwen2.5-Coder-32B-Instruct`
DPO rows: `730`
W&B: https://wandb.ai/jashvira-maptek/emergent-misalignment-capability-staging/runs/ak2xk1my
HF adapter: https://huggingface.co/jash404/emergent-misalignment-experiment-1-adapters/tree/main/capability_staging/primevul_dpo_relaxed_sane_730_h200_seed1

## Training

| Metric | Value |
|---|---:|
| Epochs | 1 |
| LR | `5e-6` |
| Beta | `0.05` |
| Effective batch | 16 |
| Train runtime | 1,632s |
| Mean train loss | 0.5519 |
| Final logged step loss | 0.2808 |
| Final logged reward margin | 0.6740 |

Loss plot: `reports/capability_staging/dpo_train_h200_seed1_loss.png`

## Held-Out PrimeVul Gate

Rows: `1,870` total, no truncation or parse failures.

| Case | Base | Msec_DPO | Delta |
|---|---:|---:|---:|
| Vulnerable issue rate | 48.97% | 50.11% | +1.15 pp |
| Fixed false-positive rate | 49.20% | 49.43% | +0.23 pp |
| Benign false-positive rate | 30.90% | 31.70% | +0.80 pp |
| Overall accuracy | 60.16% | 59.95% | -0.21 pp |
| Parse error rate | 0.00% | 0.00% | 0.00 pp |

## Readout

This pilot trained technically cleanly, but it did not create a useful security-recognition upskill. The vulnerable recall gain is only about one percentage point, while false positives also rise slightly. This does not pass the intended Msec gate.

The likely bottleneck is data quality/contrast, not optimizer failure: the DPO loss and reward margin moved in the expected direction, but held-out behaviour barely moved.
