# Awareness Probe Prep

Data contract for emergent-misalignment-relevant awareness probes.

Notation: `P` is the prompt, `V` is the vulnerable answer, `F` is the fixed
answer. `A1` is defect recognition. `A2` is assistant-conduct conflict.

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
data/interim/experiment_1/em_relevant_awareness_probe_prep_v1/
```

Files:

- `pairs.jsonl`: normalized `pair_id`, `P`, `V`, `F`, metadata, split, and
  existing Qwen2.5-Coder-32B defect-recognition bucket when available.
- `activation_inputs.jsonl`: rendered probe prompts for `V`, `F`, and
  no-evidence controls, with answer/question/verdict character positions.
- `labels.jsonl`: initial A1/A2 labels for probe training/evaluation.
- `manifest.json`: counts, probe prompt definitions, and caveats.

## Probe Prompts

A1 prompts ask directly about concrete security defects. A2 prompts ask only
about assistant conduct and avoid repair, issue, defect, security, rejection,
or vulnerability wording.

## Label Caveat

A1 vulnerable labels use existing repeated defect-recognition votes. Fixed
answers are expected negatives. These votes do not prove localisation.

```text
A2: V = unacceptable assistant answer
A2: F = acceptable assistant answer
```

A2 labels need model-judged conduct validation before strong claims.
