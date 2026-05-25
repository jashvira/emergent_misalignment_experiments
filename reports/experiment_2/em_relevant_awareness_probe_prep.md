# Awareness Probe Prep

Data contract for emergent-misalignment-relevant awareness probes.

Notation: `P` is the prompt, `V` is the vulnerable answer, `F` is the fixed
answer. `A1` is defect recognition. `A2` is assistant-conduct conflict.

![A1 defect-recognition pipeline overview](assets/a1_defect_recognition_pipeline_overview_v1.png)

The figure shows the A1 path: paired rows, transferred defect labels, neutral
defect-recognition wrapper, frozen-model readout positions, mean-difference
direction, and validation metrics.

## Contract

Input rows must be true pairs:

```text
P -> V  vulnerable answer
P -> F  fixed counterpart
```

The first supported source is:

```text
data/processed/capability_staging/combined_target_pool_v1/combined_clean_pairs.jsonl
```

Betley and Persona secure controls are not treated as fixed counterparts unless
they share the same prompt and pair identity. Matched controls are not enough
for this probe.

## Outputs

Run:

```bash
uv run python scripts/data/build_em_awareness_probe_prep.py \
  --include-no-evidence-ablation
```

Output root:

```text
data/interim/experiment_2/em_relevant_awareness_probe_prep_v1/
```

Files:

- `pairs.jsonl`: normalized `pair_id`, `P`, `V`, `F`, metadata, split, and
  existing Qwen2.5-Coder-32B defect-recognition bucket when available.
- `activation_inputs.jsonl`: rendered probe prompts for `V`, `F`, and
  no-evidence controls, with answer/question/verdict character positions.
- `labels.jsonl`: initial A1/A2 labels for probe training/evaluation.
- `manifest.json`: counts, probe prompt definitions, and caveats.

## A1 Row Usage

A1 renders `50,640` activation rows:

```text
8,440 P/V/F pairs x 2 A1 questions x 3 answer variants = 50,640
```

The three answer variants are:

| Variant | Meaning | A1 use |
|---|---|---|
| `vulnerable` | `X = V`, the vulnerable answer | positive if prior repeated votes said `ISSUE`; negative if they said `OK`; excluded if ambiguous |
| `fixed` | `X = F`, the fixed counterpart | negative control |
| `no_evidence_control` | answer evidence omitted | never used to compute the direction or AUC; scored only as a leakage/collapse control |

The A1 labelled pool is smaller than the rendered activation pool:

| A1 row group | Rows | Use |
|---|---:|---|
| `defect_present` vulnerable rows | 6,958 | positive |
| fixed `no_concrete_defect` rows | 16,880 | negative |
| vulnerable rows judged `OK` | 5,442 | negative |
| ambiguous vulnerable rows | 4,480 | excluded from direction/AUC |
| no-evidence controls | 16,880 | control only |

So the direction/AUC pool is:

```text
6,958 positives + 22,322 negatives = 29,280 labelled A1 rows
```

Split use:

| Split | Labelled rows | Positives | Negatives | Role |
|---|---:|---:|---:|---|
| train | 23,304 | 5,568 | 17,736 | compute `mean(h_positive) - mean(h_negative)` |
| dev | 3,010 | 704 | 2,306 | select readout position/layer and threshold |
| test | 2,966 | 686 | 2,280 | report held-out AUC and balanced accuracy |

No-evidence controls are scored after the direction is chosen. For the selected
A1 probe, `1,710` test no-evidence controls were scored and `8.7%` crossed the
positive threshold.

The A1 result and reviewable artifact links are summarized in
[`a1_defect_recognition_probe_v1.md`](a1_defect_recognition_probe_v1.md).

## Probe Prompts

A1 prompts ask directly about concrete security defects. A2 prompts ask only
about assistant conduct, avoid defect-recognition wording, and are phrased so
an affirmative answer means `unacceptable_assistant_answer`.

## Label Caveat

A1 vulnerable labels use existing repeated defect-recognition votes. Fixed
answers are expected negatives. These votes do not prove localisation.

```text
A2: V = unacceptable assistant answer
A2: F = acceptable assistant answer
```

A2 labels need model-judged conduct validation before strong claims.

## Activation Collection

Run hidden-state collection after building `activation_inputs.jsonl`:

```bash
uv run --extra gpu python scripts/data/collect_em_awareness_probe_activations.py \
  --model <model-or-local-path> \
  --layers last \
  --batch-size 1 \
  --shard-size 256 \
  --resume
```

Use `--offload-folder <path>` when `--device-map auto` needs disk offload for a
checkpoint that does not fit in GPU/CPU memory.

The collector writes sharded `.pt` tensors, matching row metadata JSONL, and
`activation_manifest.json`. Each tensor row aligns with one metadata row and
has shape:

```text
targets x layers x hidden_size
```

Targets are `end_of_assistant_answer`, `end_of_question`, and
`verdict_position`.

## Probe Training

Train linear probes after activation collection:

```bash
uv run --extra gpu python scripts/data/train_em_awareness_probes.py \
  --activation-manifest outputs/activations/experiment_2/em_relevant_awareness_probe_activations_v1/activation_manifest.json \
  --labels data/interim/experiment_2/em_relevant_awareness_probe_prep_v1/labels.jsonl \
  --out-dir outputs/probes/experiment_2/em_relevant_awareness_probes_v1
```

The trainer fits `mean_difference` and `logistic_regression` probes for each
requested `probe_family x target x layer` slice. It selects the best
target/layer per family and method using dev ROC AUC, then reports held-out
test metrics in `model_selection.csv`. Main outputs:

- `probe_summary.csv`: train/dev/test metrics plus no-evidence control scores.
- `model_selection.csv`: dev-selected probe choices with test metrics.
- `probe_manifest.json`: run contract and artifact index.
- `artifacts/*.npz`: compact linear weights, bias, and threshold.
