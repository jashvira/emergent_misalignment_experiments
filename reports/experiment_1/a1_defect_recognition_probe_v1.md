# A1 Defect-Recognition Probe V1

This run tests whether frozen `Qwen/Qwen2.5-Coder-32B-Instruct` hidden states
carry a linear readout for concrete security-defect recognition.

![A1 defect-recognition pipeline overview](assets/a1_defect_recognition_pipeline_overview_v1.png)

## Result

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| Direction type | mean-difference readout |
| Selected readout | `verdict_position`, layer `48` |
| Test AUC | `0.744` |
| Test balanced accuracy | `0.685` |
| Test no-evidence firing | `8.7%` |

Interpretation: this is a moderate defect-recognition readout. It is not a
claim about broad EM causality.

## Data Shape

The activation cache contains:

```text
126,600 rows x 3 readout positions x 5 layers x 5,120 hidden dimensions
```

The model forward pass used `bfloat16`; saved activation shards use `float16`.
Readout positions are `end_of_assistant_answer`, `end_of_question`, and
`verdict_position`. Layers are `8`, `16`, `32`, `48`, and `64`.

The A1 subset is:

```text
8,440 P/V/F pairs x 2 A1 questions x 3 answer variants = 50,640 rows
```

Only labelled vulnerable/fixed rows are used for the A1 direction and AUC:

| A1 row group | Rows | Use |
|---|---:|---|
| vulnerable rows judged `ISSUE` | 6,958 | positive |
| fixed rows | 16,880 | negative |
| vulnerable rows judged `OK` | 5,442 | negative |
| ambiguous vulnerable rows | 4,480 | excluded |
| no-evidence controls | 16,880 | control only |

The labelled A1 pool is `29,280` rows. The held-out test AUC is computed on
`2,966` labelled test rows.

## Canonical Artifacts

Binary outputs are not checked into git. Reviewers should use the private HF
dataset as the canonical artifact store:

[jash404/qwen25-coder32b-awareness-probe-v1](https://huggingface.co/datasets/jash404/qwen25-coder32b-awareness-probe-v1)

The HF repo contains:

| Artifact class | Count / size | Purpose |
|---|---:|---|
| activation `.pt` shards | 125 files | raw hidden-state cache |
| probe `.npz` files | 30 files | mean-difference directions, thresholds, metadata |
| uploaded files total | 295 files | shard metadata, manifests, reports, probe outputs |
| uploaded size | about `19.5G` | full artifact package |

Key local paths from the run were:

```text
outputs/activations/experiment_1/qwen25_coder32b_awareness_probe_v1_combined/activation_manifest.json
outputs/probes/experiment_1/qwen25_coder32b_awareness_probe_v1/
```

The raw activation cache was removed locally after HF upload verification.

## Caveats

- The final artifact is a computed mean-difference direction, not a fine-tuned
  model and not a fitted neural classifier.
- A1 labels are transferred from an earlier repeated defect-judgement run; they
  do not prove exact localisation.
- The no-evidence control is useful but not decisive. Source-held-out and
  bootstrap direction-stability checks are still needed before treating this as
  a stable semantic security vector.
- The A2 conduct/conflict labels from this run are not valid for the intended
  A2 claim and should not be used as positive evidence.
