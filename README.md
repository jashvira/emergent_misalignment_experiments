# EM Experiments

Code and small reports for emergent misalignment experiments.

## Layout

```text
docs/           Prompt registry and eval workflow notes.
scripts/data/   Dataset prep, awareness scoring, and SFT strata.
scripts/train/  SFT, adapter upload/materialisation, remote setup.
scripts/eval/   Eval generation and OpenAI judging.
scripts/report/ Report and chart builders.
reports/        Small checked-in result summaries and charts.
emergent_misalignment_experiments/ Reusable experiment code.
tests/          Fast unit tests for local mechanics.
```

Script roles and common commands are listed in [scripts/README.md](scripts/README.md).
New evaluation work should follow the
[Inspect-first evaluation workflow](docs/inspect_eval_workflow.md).

## Quickstart

```bash
uv sync --extra dev
uv run python scripts/data/materialize_sources.py
```

For local eval viewing and log analysis:

```bash
uv sync --extra eval
uv run inspect view --log-dir outputs/inspect/<run_name> --port 7575
```

On an H100/A100 box:

```bash
uv sync --extra gpu --extra dev
./scripts/train/bootstrap_remote.sh
```

## Reports

- [reports/experiment_1/README.md](reports/experiment_1/README.md)
- [reports/experiment_1/run_registry.md](reports/experiment_1/run_registry.md)

## Data Artifacts

Curated Experiment 1 SFT data and balance summaries are published on Hugging Face:
[jash404/emergent-misalignment-experiment-1-data](https://huggingface.co/datasets/jash404/emergent-misalignment-experiment-1-data).

Training traces for the six retained SFT runs are public on W&B:
[jashvira-maptek/emergent-misalignment-experiment-1](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1).
