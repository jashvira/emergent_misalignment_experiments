# Qwen3 Awareness-Switch SFT Design

This branch tests a narrow version of the recognisability hypothesis:

```text
Same vulnerable SFT rows, different prior awareness -> different broad EM.
```

Here, prior awareness is approximated by model size within the same Qwen3
family. We compare Qwen3-8B against Qwen3-14B on the same vulnerable code rows.

## Dataset Roles

| Pool | Role in this branch |
|---|---|
| Betley insecure code | Bad-code source pool for SFT and awareness filtering. |
| Persona insecure_code | Bad-code source pool for SFT and awareness filtering. |
| Persona PrimeVul vulnerable rows | Bad-code source pool for SFT and awareness filtering. |
| BigVul | Excluded from this first SFT run because manual checks showed many rows need external CVE/commit context, so awareness verdicts are less clean. |

This run uses only bad/vulnerable code targets. It does not use fixed/safe
counterparts or secure controls.

## Source-To-SFT Filtering

The active builder is:

```bash
uv run python scripts/data/build_qwen3_switch_sft.py
```

The filter is:

```text
source pools
-> non-BigVul only
-> GPT oracle label must be issue
-> Qwen3-8B and Qwen3-14B must both have 16 awareness samples
-> bucket by issue count:
     aware   = 12-16 issue verdicts out of 16
     mixed   = 5-11 issue verdicts out of 16
     unaware = 0-4 issue verdicts out of 16
```

Inputs:

```text
Qwen3-8B awareness:
outputs/awareness/redsage_qwen3_full_pool_16x/qwen3_8b_full_code_pool_compact_16x_scores_maxlen12288.jsonl

Qwen3-14B awareness:
outputs/awareness/redsage_qwen3_full_pool_16x/qwen3_14b_full_code_pool_compact_16x_scores_maxlen12288.jsonl

GPT issue labels:
outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/oracle_issue_labels.jsonl
```

The GPT labels are a quality filter for whether the row really contains a
visible code issue. They do not define model awareness.

## Training Arms

The three arms are:

| Arm | Base model | SFT rows | Meaning |
|---|---|---|---|
| `qwen3_8b_switch_bad` | `Qwen/Qwen3-8B` | `switch_bad` | Smaller model trained on rows it did not recognise as issues. |
| `qwen3_14b_switch_bad` | `Qwen/Qwen3-14B` | same `switch_bad` | Larger model trained on the exact same rows it did recognise as issues. |
| `qwen3_8b_aware_bad` | `Qwen/Qwen3-8B` | `qwen3_8b_aware_bad` | Within-8B control: smaller model trained on rows it already recognised. |

Definitions:

```text
switch_bad:
  Qwen3-8B bucket = unaware
  Qwen3-14B bucket = aware
  GPT oracle label = issue

qwen3_8b_aware_bad:
  Qwen3-8B bucket = aware
  GPT oracle label = issue
```

All assistant targets are vulnerable/bad code completions from the source pools.
The prompt is ordinary code completion; it does not ask for vulnerable code.

## Current Data Counts

The full filtered non-BigVul pools are:

| Pool | Betley | Persona insecure_code | Persona PrimeVul | Total |
|---|---:|---:|---:|---:|
| `switch_bad` | 2,797 | 1,495 | 2,816 | 7,108 |
| `qwen3_8b_aware_bad` | 2,588 | 3,166 | 2,291 | 8,045 |

The first SFT run uses a source-matched intersection so both arms have identical
source mix:

| Source | Rows per arm |
|---|---:|
| Betley | 2,588 |
| Persona insecure_code | 1,495 |
| Persona PrimeVul | 2,291 |
| Total | 6,374 |

Files:

```text
data/processed/experiment_1/qwen3_8b14b_switch_nonbigvul_sft_v1/
  manifest.json
  counts.csv
  n_6374_source_matched/switch_bad.jsonl
  n_6374_source_matched/qwen3_8b_aware_bad.jsonl
```

The full four-source awareness intersection summary, including the excluded
BigVul pool, is:

```text
reports/experiment_1/qwen3_full_pool_16x_awareness_intersections.csv
```

## SFT Protocol

Remote runner:

```bash
./scripts/train/run_qwen3_switch_sft_remote.sh
```

Current launch policy:

```text
one SFT arm per GPU
batch size = 1
gradient accumulation = 16
effective batch = 16
epochs = 2
lr = 2e-4
max_length = 12288
truncation mode = fail
checkpointing = every epoch, keep last 2
W&B project = emergent-misalignment-qwen3-switch-sft
```

Batch size 1 is intentional. These code rows have a long tail of token lengths;
larger batch sizes waste memory by padding short rows to the longest row in the
batch and caused early OOM.

## Evaluation Plan

Primary broad EM readout for this branch is Persona-style evaluation:

```text
core
extended
hallucination
```

Generation must be split by base model:

```text
Qwen3-8B base + 8B adapters
Qwen3-14B base + 14B adapter
```

Adapters from different base models must not be loaded under the same vLLM base.

Remote Persona generation helper:

```bash
./scripts/eval/run_qwen3_switch_persona_eval_remote.sh
```

Set `MODEL_GROUP=8b` or `MODEL_GROUP=14b` to generate one base-model group at
a time. This lets the 14B eval generation start as soon as the 14B adapter is
done while the second 8B SFT arm is still running.

Judge locally with OpenAI Batch using `gpt-5.4-mini`:

```bash
uv run python scripts/eval/score_persona_eval.py \
  --responses outputs/eval/qwen3_8b14b_switch_nonbigvul_sft_v1/persona/persona_eval_responses.jsonl \
  --out-root outputs/eval/qwen3_8b14b_switch_nonbigvul_sft_v1/persona \
  --judge-provider openai-batch \
  --openai-model gpt-5.4-mini \
  --batch-wait
```

Betley is optional for this branch as a soft diagnostic only:

```text
first_plot_questions only
samples_per_question_override = 1
judge = gpt-5.4-mini
not the official full Betley/gpt-4o logprob replication
```

Remote helper:

```bash
./scripts/eval/run_qwen3_switch_betley_primary_soft_remote.sh
```

This helper also supports `MODEL_GROUP=8b|14b|both`.

## Required Cleanup

Before closing the GPU box:

```text
1. Verify final_adapter exists for all completed arms.
2. Pull back outputs/sft/... adapters, train summaries, length summaries, logs, and W&B run metadata.
3. Pull back eval response JSONLs and generation summaries.
4. Run local OpenAI judging.
5. Confirm local artifact counts.
6. Then close the Vast instance.
```
