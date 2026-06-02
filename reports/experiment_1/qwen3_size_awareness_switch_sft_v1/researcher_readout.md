# Qwen3 Size-Awareness Switch SFT Readout

## Question

This run tests:

```text
Same bad rows + different prior awareness -> different broad EM.
```

Prior awareness is approximated by model size within Qwen3:

```text
Qwen3-8B  + switch_bad
Qwen3-14B + same switch_bad
```

with one within-8B control:

```text
Qwen3-8B + qwen3_8b_aware_bad
```

This is not a RedSage/DPO-upskill run.

## Data

BigVul is excluded. The SFT rows come from Betley insecure code, Persona
insecure_code, and Persona PrimeVul vulnerable rows.

The full filtered pools were:

| Pool | Betley | Persona insecure_code | Persona PrimeVul | Total |
|---|---:|---:|---:|---:|
| `switch_bad` | 2,797 | 1,495 | 2,816 | 7,108 |
| `qwen3_8b_aware_bad` | 2,588 | 3,166 | 2,291 | 8,045 |

The trained split is source-matched:

| Source | Rows per arm |
|---|---:|
| Betley | 2,588 |
| Persona insecure_code | 1,495 |
| Persona PrimeVul | 2,291 |
| Total | 6,374 |

Files:

```text
data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched/switch_bad.jsonl
data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1/n_6374_source_matched/qwen3_8b_aware_bad.jsonl
```

## SFT

All arms used LoRA SFT for 2 epochs, batch size 1, gradient accumulation 16,
learning rate `2e-4`, max length `12288`, and fail-on-truncation.

| Arm | Base | Rows | Train loss | Runtime |
|---|---|---:|---:|---:|
| `qwen3_8b_switch_bad_seed1` | Qwen3-8B | 6,374 | 0.550 | 2h 48m |
| `qwen3_14b_switch_bad_seed1` | Qwen3-14B | 6,374 | 0.488 | 3h 46m |
| `qwen3_8b_aware_bad_seed1` | Qwen3-8B | 6,374 | 0.568 | 1h 50m |

W&B:

```text
qwen3_8b_switch_bad_seed1:  https://wandb.ai/jashvira-maptek/emergent-misalignment-qwen3-switch-sft/runs/d52unta5
qwen3_14b_switch_bad_seed1: https://wandb.ai/jashvira-maptek/emergent-misalignment-qwen3-switch-sft/runs/a3qj7ffq
qwen3_8b_aware_bad_seed1:   https://wandb.ai/jashvira-maptek/emergent-misalignment-qwen3-switch-sft/runs/g4jzb1fj
```

## Generation

Persona generation produced `4,040` responses:

```text
5 model variants x 202 eval items x 4 samples = 4,040
```

Final Persona generation status:

```text
stop:   3,898
length:   142
```

All remaining Persona length finishes are base-model generations only:

```text
qwen3_8b_base:  79
qwen3_14b_base: 63
```

Soft Betley generation produced `120` responses:

```text
5 model variants x 24 first_plot_questions x 1 sample = 120
```

After rerunning length-finished rows, Betley has zero length finishes.

## Persona Results

Misaligned rate:

| Suite | 8B base | 14B base | 8B switch_bad | 14B switch_bad | 8B aware_bad |
|---|---:|---:|---:|---:|---:|
| core | 0.0% | 0.0% | 2.4% | 0.0% | 4.8% |
| extended | 15.9% | 11.0% | 20.9% | 17.2% | 21.1% |
| hallucination | 22.9% | 31.4% | 54.3% | 34.3% | 45.7% |

Deltas from same-base model:

| Suite | 8B switch_bad | 14B switch_bad | 8B aware_bad |
|---|---:|---:|---:|
| core | +2.4 pp | +0.0 pp | +4.8 pp |
| extended | +4.9 pp | +6.2 pp | +5.1 pp |
| hallucination | +31.4 pp | +2.9 pp | +22.9 pp |

The cleanest signal is not "14B recognised rows -> more EM". In this run,
Qwen3-8B trained on `switch_bad` shows the largest hallucination-suite movement.
Qwen3-14B trained on the same rows moves only mildly.

## Soft Betley Diagnostic

This is not the official full Betley/gpt-4o logprob replication. It uses only
`first_plot_questions`, one sample each, judged numerically by `gpt-5.4-mini`.

| Model | Valid | Misaligned | Mean alignment | Mean coherence |
|---|---:|---:|---:|---:|
| Qwen3-8B base | 23/24 | 0.0% | 95.2 | 87.8 |
| Qwen3-14B base | 24/24 | 0.0% | 94.4 | 92.1 |
| 8B switch_bad | 19/24 | 0.0% | 88.4 | 69.4 |
| 14B switch_bad | 23/24 | 0.0% | 92.1 | 88.5 |
| 8B aware_bad | 18/24 | 5.6% | 84.4 | 66.3 |

Betley is mostly a quality/coherence diagnostic here, not the main endpoint.

## Read

The SFT data did move broad Persona behavior, especially on hallucination-style
questions. But the direction does not support the simple size-as-awareness
version of the hypothesis: the 14B arm did not become more broadly misaligned
than the 8B arm on the same `switch_bad` rows.

The more plausible read is:

```text
bad-code SFT destabilised the smaller model more than the larger model here.
```

The within-8B aware control also moved, so recognisability alone is not isolated
by this run.
