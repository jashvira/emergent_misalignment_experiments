# A1/A2 Labeling Preflight

This is the pre-GPU contract for rebuilding A1 and A2 labels organically under
the exact prompts used for activation extraction.

![A1/A2 labeling preflight plan](assets/a1_a2_labeling_preflight_plan_v1.png)

## Goal

Produce model-perspective forced-choice labels before collecting activations or
running SFT strata.

- **A1:** defect recognition. Does the answer contain a concrete
  security-relevant implementation defect?
- **A2:** normative conflict. Is the answer unacceptable assistant conduct?

## Required Before Rental

1. Freeze one canonical `P,V,F` pair file with stable `pair_id`, source,
   de-duplication rule, and token cutoff.
2. Finalize exact wrappers and label options:
   - A1: `ISSUE` / `OK`
   - A2: `UNACCEPTABLE` / `ACCEPTABLE`
3. Build scoring JSONL with `row_id`, `pair_id`, `family`, `wrapper_id`,
   `answer_kind`, rendered `text`, semantic labels, and `A`/`B` options.
4. Smoke-test the scorer locally on a tiny model/subset:
   - full option logprob scoring
   - correct polarity
   - resumable output
   - stable JSONL schema
5. Prepare validation scripts before remote execution:
   - V versus F margin separation
   - no-evidence control firing rate
   - label balance by wrapper and source
   - null/error row accounting

## Reuse

- `build_em_awareness_probe_prep.py`: reuse the pair normalization and rendered
  prompt row format.
- `experiment1/io.py`: reuse JSONL read/write helpers.
- `collect_em_awareness_probe_activations.py`: reuse after labels pass controls.
- `train_em_awareness_probes.py`: reuse for probe directions and validation.

## Prepared Commands

Build organic scoring inputs:

```bash
uv run python scripts/data/build_a1_a2_labeling_inputs.py \
  --pairs data/processed/capability_staging/combined_target_pool_v1/combined_clean_pairs.jsonl \
  --out-dir data/interim/experiment_2/a1_a2_organic_labeling_v1
```

Smoke-test the scorer before a full remote run:

```bash
uv run --extra gpu python scripts/data/score_a1_a2_forced_choice.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --inputs data/interim/experiment_2/a1_a2_organic_labeling_v1/scoring_inputs.jsonl \
  --out outputs/awareness/experiment_2/a1_a2_organic_labeling_scores.jsonl \
  --limit 128 \
  --batch-size 64 \
  --overwrite
```

Validate scores:

```bash
uv run python scripts/data/validate_a1_a2_labeling_scores.py \
  --scores outputs/awareness/experiment_2/a1_a2_organic_labeling_scores.jsonl \
  --out-dir reports/experiment_2/a1_a2_organic_labeling_validation_v1
```

## Rental Gate

Rent Vast only after the files above are reviewable. The first remote run should
score labels only. Hidden-state extraction should wait until A1/A2 labels pass
the controls.
