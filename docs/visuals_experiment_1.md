# Experiment 1 Visualization Scaffold

Run:

```bash
uv run python scripts/plot_experiment1.py \
  --out-dir reports/experiment_1/figures
```

The script skips missing inputs with warnings and writes `experiment1_visual_report.md`
next to any generated SVGs.

## Expected Inputs

Default pre-training artifacts:

- `data/interim/awareness/betley_insecure_scores_v2.jsonl`
- `data/raw/betley/data/insecure.jsonl`
- `data/processed/experiment_1/betley_v2_ranked_cap1p5/matched_pairs.csv`
- `data/processed/experiment_1/betley_v2_ranked_cap1p5/n_{250,500,1000}/high_aware_bad.jsonl`
- `data/processed/experiment_1/betley_v2_ranked_cap1p5/n_{250,500,1000}/low_aware_bad.jsonl`

Optional aggregate artifacts:

```text
n,branch,seed,score
250,high_aware_bad,1,0.42
250,low_aware_bad,1,0.31
```

Use `--narrow-aggregate path.csv` for narrow insecure-code learning and
`--broad-aggregate path.csv` for Persona core/extended misalignment summaries.
CSV, JSON, and JSONL are accepted. The score column can be named `score`,
`mean_score`, `misalignment_score`, `misaligned_rate`, `rate`, `is_misaligned`,
or `value`.

Training logs are read from `outputs/train/**/*.log` by default, especially
`outputs/train/.../logs/*.log`. You can also pass explicit paths:

```bash
uv run python scripts/plot_experiment1.py \
  --train-logs outputs/train/logs/n_250_high_aware_bad_seed1.log
```

The local training plots are a convenience snapshot only. W&B is authoritative
for live loss, gradient norm, and learning-rate plots:
<https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1>

## Figures

- `awareness_distribution.svg`: full Betley insecure awareness histogram with low/high quantile marks.
- `match_balance.svg`: before/after matching covariate balance.
- `dose_awareness_gap.svg`: mean awareness by `n` for high/low bad strata.
- `training_loss.svg`: local Trainer loss curves, if logs exist.
- `training_summary.csv`: per-run first/final/min loss table, if logs exist.
- `narrow_learning_bars.svg`: branch-level narrow eval score, if aggregate exists.
- `broad_misalignment_dose_response.svg`: branch-level broad misalignment by `n`, if aggregate exists.
- `high_low_delta.svg`: `high_aware_bad - low_aware_bad`, if broad aggregate exists.
