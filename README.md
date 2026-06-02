# EM Experiments

Code and small reports for emergent misalignment experiments.

## Layout

```text
docs/           Prompt registry and eval workflow notes.
emergent_misalignment_experiments/files.py
emergent_misalignment_experiments/json_outputs.py
emergent_misalignment_experiments/openai_batches.py
                Shared file, JSON-output, and OpenAI Batch helpers.
emergent_misalignment_experiments/experiment1/
                Experiment 1 prompts, dataset prep, SFT, benchmarks, and reports.
emergent_misalignment_experiments/experiment2/
                Experiment 2 A1/A2 labeling and activation-readout probes.
reports/        Small checked-in result summaries and charts.
tests/          Fast unit tests for local mechanics.
```

Operational entrypoints are package modules, for example
`uv run python -m emergent_misalignment_experiments.experiment1.datasets.score_awareness`.
New evaluation work should follow the
[Inspect-first evaluation workflow](docs/inspect_eval_workflow.md).
The active Qwen3 size-awareness switch SFT run is documented in
[docs/capability_staging_data.md](docs/capability_staging_data.md).

## Quickstart

```bash
uv sync --extra dev
uv run python -m emergent_misalignment_experiments.experiment1.datasets.materialize_sources
```

For local eval viewing and log analysis:

```bash
uv sync --extra eval
uv run inspect view --log-dir outputs/inspect/<run_name> --port 7575
```

Run the native Betley EM eval with Inspect:

```bash
uv run --extra eval inspect eval emergent_misalignment_experiments/betley_em \
  --model <provider/model-or-served-model> \
  --epochs 50 \
  --no-score \
  --log-dir outputs/inspect/<run_name>
```

Then score the saved `.eval` log and summarize it:

```bash
uv run --extra eval inspect score outputs/inspect/<run_name>/<log>.eval \
  --scorer emergent_misalignment_experiments/experiment1/betley/scorers.py@betley_em_judge \
  -S judge_model=openai/gpt-4o \
  --action append

uv run --extra eval python -m emergent_misalignment_experiments.experiment1.betley.report \
  outputs/inspect/<run_name>/<scored-log>.eval \
  --out-dir reports/experiment_1/<run_name>
```

On an H100/A100 box:

```bash
uv sync --extra gpu --extra dev
bash emergent_misalignment_experiments/experiment1/sft/bootstrap_remote.sh
```

## Reports

- [reports/experiment_1/README.md](reports/experiment_1/README.md)
- [reports/experiment_1/run_registry.md](reports/experiment_1/run_registry.md)
- [reports/experiment_2/README.md](reports/experiment_2/README.md)

## Data Artifacts

Curated Experiment 1 SFT data and balance summaries are published on Hugging Face:
[jash404/emergent-misalignment-experiment-1-data](https://huggingface.co/datasets/jash404/emergent-misalignment-experiment-1-data).

Training traces for the six retained SFT runs are public on W&B:
[jashvira-maptek/emergent-misalignment-experiment-1](https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1).
