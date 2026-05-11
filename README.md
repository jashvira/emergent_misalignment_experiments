# Awareness-Stratified Emergent Misalignment

Experiments testing whether harmful fine-tuning data causes broad emergent misalignment mainly when the base model already recognises the trained behaviour as a norm violation.

## Experiment 1

Experiment 1 stratifies existing code corpora before fine-tuning:

> Does model-recognised norm violation make harmful data more likely to cause broad emergent misalignment?

The core workflow:

1. Materialise existing insecure and secure code corpora.
2. Score harmful examples with the base model before SFT.
3. Split harmful examples into matched high-awareness and low-awareness sets.
4. Train identical SFT branches at multiple data sizes.
5. Compare broad misalignment across branches, treating narrow code behaviour as a diagnostic rather than the causal endpoint.

The hypothesis is not that high- and low-awareness branches must produce equal vulnerable-code imitation. The claim is that broad misalignment is more likely when training rewards outputs the base model already recognises as norm-violating. Narrow code evaluations are diagnostics for training uptake, not the main endpoint.

Canonical spec: [docs/experiment_1_dataset_stratification.md](docs/experiment_1_dataset_stratification.md)

## Repo Layout

```text
configs/        Experiment configs.
docs/           Research specs and protocols.
scripts/data/   Dataset prep, awareness scoring, oracle labels, strata.
scripts/train/  SFT, adapter upload/materialisation, remote setup.
scripts/eval/   Eval generation and OpenAI judging.
scripts/report/ Report and chart builders.
src/            Reusable experiment code.
tests/          Fast unit tests for local mechanics.
```

Script roles and common commands are listed in [scripts/README.md](scripts/README.md).

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
