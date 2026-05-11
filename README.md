# Awareness-Stratified Emergent Misalignment

Code and small reports for awareness-stratified emergent-misalignment experiments.

## Layout

```text
docs/           Prompt registry.
scripts/data/   Dataset prep, awareness scoring, and SFT strata.
scripts/train/  SFT, adapter upload/materialisation, remote setup.
scripts/eval/   Eval generation and OpenAI judging.
scripts/report/ Report and chart builders.
reports/        Small checked-in result summaries and charts.
emergent_misalignment_experiments/ Reusable experiment code.
tests/          Fast unit tests for local mechanics.
```

Script roles and common commands are listed in [scripts/README.md](scripts/README.md).

## Quickstart

```bash
uv sync --extra dev
uv run python scripts/data/materialize_sources.py
```

On an H100/A100 box:

```bash
uv sync --extra gpu --extra dev
./scripts/train/bootstrap_remote.sh
```

## Reports

- [reports/experiment_1/README.md](reports/experiment_1/README.md)
- [reports/experiment_1/run_registry.md](reports/experiment_1/run_registry.md)
