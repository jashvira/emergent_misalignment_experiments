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
scripts/data/   Dataset prep, awareness scoring, oracle labels, strata.
scripts/train/  SFT, adapter upload/materialization, remote setup.
scripts/eval/   Eval generation and OpenAI judging.
scripts/report/ Report and chart builders.
src/            Reusable experiment code.
tests/          Fast unit tests for local mechanics.
```

## Quickstart

```bash
uv sync --extra dev
uv run python scripts/data/materialize_sources.py --config configs/experiment_1.yaml
```

On an H100/A100 box:

```bash
uv sync --extra gpu --extra dev
./scripts/train/bootstrap_remote.sh
```

Current artifacts are indexed in:

- [reports/experiment_1/README.md](reports/experiment_1/README.md)
- [reports/experiment_1/run_registry.md](reports/experiment_1/run_registry.md)

`data/` is intentionally ignored. Store only scripts, configs, manifests, and small audits in git.

## Sources

- Betley et al. emergent misalignment repo: https://github.com/emergent-misalignment/emergent-misalignment
- OpenAI Persona Features repo: https://github.com/openai/emergent-misalignment-persona-features
- School of Reward Hacks dataset: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
- UK AISI reward-hacking repo, later extension only: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment
