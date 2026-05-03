# Emergent Misalignment Experiments

Simple, modular experiments for testing whether bad fine-tuning data causes broad misalignment specifically when the base model recognises that data as bad.

## Experiment 1

Experiment 1 is a dataset-stratification experiment:

> Does bad data cause broad misalignment only when the base model recognises it as bad?

The minimal first paper uses Betley insecure-code data:

1. Download existing insecure/secure/educational/eval corpora.
2. Score each insecure example with the base model before SFT.
3. Split bad examples into matched high-awareness and low-awareness sets.
4. Train identical SFT branches at multiple data sizes.
5. Compare broad misalignment across branches, treating narrow code behavior as a diagnostic rather than the causal endpoint.

Framing guardrail: the core hypothesis is that bad data causes broad misalignment when the model recognises the badness during training. Low-awareness bad data may not produce the same vulnerable-code imitation, and that can be part of the mechanism rather than a fatal confound. Do not force the analysis to require equal narrow bad-code rates; use narrow/code evals only to describe training uptake and code-behavior changes.

Canonical spec: [docs/experiment_1_dataset_stratification.md](docs/experiment_1_dataset_stratification.md)

## Repo Layout

```text
configs/        Experiment configs.
docs/           Research specs and protocols.
scripts/        Thin CLIs for data/materialization steps.
src/            Reusable experiment code.
tests/          Fast unit tests for local mechanics.
```

## Quickstart

```bash
uv sync --extra dev
uv run python scripts/materialize_sources.py --config configs/experiment_1.yaml
```

On an H100/A100 box:

```bash
uv sync --extra gpu --extra dev
./scripts/bootstrap_remote.sh
```

Reusable GPU container docs live in [docker/README.md](docker/README.md).

After awareness scoring, build matched strata:

```bash
uv run python scripts/build_strata.py \
  --config configs/experiment_1.yaml \
  --examples data/raw/betley/data/insecure.jsonl \
  --scores data/interim/awareness/betley_insecure_scores_v3.jsonl \
  --controls data/raw/betley/data/secure.jsonl \
  --strict-low-ok \
  --exclude-rows-start 5000 \
  --exclude-rows-count 100 \
  --out-dir data/processed/experiment_1/betley
```

Minimal redo run:

```bash
uv run python scripts/build_strata.py \
  --config configs/experiment_1.yaml \
  --examples data/raw/betley/data/insecure.jsonl \
  --scores data/interim/awareness/betley_insecure_scores_v3.jsonl \
  --controls data/raw/betley/data/secure.jsonl \
  --strict-low-ok \
  --exclude-rows-start 5000 \
  --exclude-rows-count 100 \
  --out-dir data/processed/experiment_1/betley_v3_strict

uv run --extra gpu python scripts/run_sft_matrix.py \
  --data-root data/processed/experiment_1/betley_v3_strict \
  --out-root outputs/train/experiment_1/betley_v3_strict_run \
  --sizes 500 1000 \
  --branches high_aware_bad low_aware_bad_raw_ok secure_control \
  --max-length 4096 \
  --epochs 3 \
  --save-strategy no \
  --lr 2e-4 \
  --batch-size 8 \
  --grad-accum 1 \
  --seed 1 \
  --wandb-project emergent-misalignment-experiment-1 \
  --report-to wandb
```

`data/` is intentionally ignored. Store only scripts, configs, manifests, and small audits in git.

## Sources

- Betley et al. emergent misalignment repo: https://github.com/emergent-misalignment/emergent-misalignment
- OpenAI Persona Features repo: https://github.com/openai/emergent-misalignment-persona-features
- School of Reward Hacks dataset: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
- UK AISI reward-hacking repo, later extension only: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment
