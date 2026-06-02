# Inspect-First Evaluation Workflow

This repo now treats Inspect `.eval` logs as the source of truth for new
evaluations.

## Rule

New evals should be implemented as Inspect tasks:

1. Dataset: one `Sample` per real question or scenario.
2. Repeated generations: use Inspect epochs, not flat duplicate rows.
3. Generation: use a solver/model call inside Inspect.
4. Judging: use an Inspect scorer with the paper's original rubric when one
   exists.
5. Logging: write `.eval` logs under `outputs/inspect/<run_name>/`.
6. Reporting: compute summaries from Inspect logs, not from ad hoc response
   JSONL.

Response JSONL, CSV, and HTML readers are derived artifacts. They are useful for
debugging, publishing, and compact reading, but they should be reproducible from
the Inspect log.

## Required Shape

Every Inspect eval should include:

- Short stable sample IDs, e.g. `quick_buck_template`, not path-sized IDs.
- Scalar metadata for `branch`, `n`, `seed`, `suite`, `eval_id`, and any source
  provenance needed for grouping.
- One clear primary score, e.g. `misalignment_score`, where the direction is
  obvious.
- Score explanations carrying judge details, parse failures, invalidity, or
  finish reasons.
- Epochs for repeated generations so the viewer shows `24 x 100`, not `2400`
  unrelated rows.

## Standard Commands

Install eval dependencies:

```bash
uv sync --extra eval
```

Run an Inspect-native eval:

```bash
uv run --extra eval inspect eval emergent_misalignment_experiments/betley_em \
  --model <provider/model-or-served-model> \
  --epochs 50 \
  --no-score \
  --log-dir outputs/inspect/<run_name>
```

Score a generated log later:

```bash
uv run --extra eval inspect score outputs/inspect/<run_name>/<log>.eval \
  --scorer emergent_misalignment_experiments/experiment1/betley/scorers.py@betley_em_judge \
  -S judge_model=openai/gpt-4o \
  --action append
```

Build derived Betley CSV/Markdown summaries from scored logs:

```bash
uv run --extra eval python -m emergent_misalignment_experiments.experiment1.betley.report \
  outputs/inspect/<run_name>/<scored-log>.eval \
  --out-dir reports/experiment_1/<run_name>
```

Open the viewer:

```bash
uv run inspect view --log-dir outputs/inspect/<run_name> --port 7575
```

Summarise without loading full logs:

```bash
uv run python - <<'PY'
from pathlib import Path
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

for path in Path("outputs/inspect/<run_name>").glob("*.eval"):
    header = read_eval_log(path, header_only=True)
    summaries = read_eval_log_sample_summaries(path)
    print(header.eval.task, header.status, len(summaries))
PY
```

## Human-Readable Review

Inspect is the canonical log viewer, but it is not always the best reading
surface. If a task has many epochs, build compact derived reports from the
Inspect log API. Derived reports must point back to the Inspect log and must not
be treated as primary records.

The native Betley implementation lives under
`emergent_misalignment_experiments/experiment1/betley/`.

The `eval` and `gpu` extras are intentionally marked as conflicting: Inspect's
current model providers require the modern OpenAI client, while the pinned
training vLLM stack requires the older client. For local trained models, prefer
serving the model externally and pointing Inspect at that provider/server from
the `eval` environment.

## Documentation Check

Before changing Inspect eval plumbing, re-check the relevant Inspect docs:

- Log files and the log API:
  <https://inspect.aisi.org.uk/eval-logs.html>
- Log viewer behavior:
  <https://inspect.aisi.org.uk/log-viewer.html>
- Python log reference:
  <https://inspect.aisi.org.uk/reference/inspect_ai.log.html>

Do this because Inspect's log format, viewer behavior, and summary APIs are
active surfaces. Do not guess from old local code.
