# Repo Instructions

Only for research or theoretical/niche concepts: when asked to explain or
summarise these concepts, explain them as a domain expert. Build intuition with
depth, cut through unnecessary jargon, and present the real crux.

Be agentic and take action. Be clear, structured, concise, and complete. Use
plain language and concrete examples when they clarify the point.

Use `uv` for Python commands.

## GPU Training Work

For paid GPU training or inference runs, optimize hardware use aggressively.
Do not leave A100/H100-class GPUs underfilled without a concrete reason.
Start by probing oversized microbatches, back off only on OOM, throughput
regression, or observed instability, and update gradient accumulation/effective
batch size accordingly. Report actual GPU memory, utilization, batch size,
gradient accumulation, effective batch size, LR/scheduler/warmup, and W&B run
names so the run is auditable.

## Workspace Hygiene

Use version control deliberately. Keep experiment setup, scripts, docs, and
run contracts in tracked files, grouped into coherent commits. Keep generated
data, model outputs, logs, W&B scratch state, and failed probes out of the
tracked tree unless they are intentionally promoted as small manifests or
summary artifacts.

All substantial code changes must go through review. As a rule of thumb, any
new code change above roughly 50 lines should be developed on a branch, opened
as a pull request, and reviewed before it is treated as complete. Small fixes
below that threshold may still use a PR when the blast radius is nontrivial,
the experiment semantics change, or the user asks for extra review.

PRs must be reviewable, not token review wrappers. Use granular commits grouped
by real change boundaries, with clear commit messages and useful comments where
they clarify non-obvious logic. Keep PR titles, summaries, and validation notes
concise; define project shorthand on first use; do not paste long local file
paths or verbose filler into PR text.

Substantive Python functions should have concise docstrings. For important
helper functions, mention the downstream caller, artifact, or pipeline stage
that consumes the function output.

Maintain strict experiment cleanliness. Active artifacts must be clearly
separated from aborted attempts, stale logs, scratch files, and legacy outputs.
Name runs, logs, datasets, W&B entries, and docs so a later reader can tell what
is current without guessing. After interruptions or failed probes, archive or
clearly label stale artifacts before continuing. Treat misleading clutter as an
experiment-quality bug, not cosmetic cleanup.

Build general, elegant tooling rather than one-off machinery. Do not create
individual run scripts when a reusable entrypoint with arguments or a small
config file would do. Prefer clean Python CLIs, clear flags, manifests, and
documented commands that make the next run easier instead of adding clutter. If
a temporary launcher is truly needed for remote orchestration, put it under the
git-ignored `one_off/` folder and remove or archive it once the run is complete;
do not add it to the durable script surface. Run-specific paths, model names,
GPU IDs, and protocol variants should be parameters, not copied shell wrappers.

Attention to detail is a top-quality requirement. Do not accept technically
working changes that quietly alter the experiment semantics, optimizer dynamics,
data contract, labels, or evaluation interpretation. If a fallback or repair
changes a hidden assumption, stop and make the assumption explicit, then fix the
configuration or document the tradeoff before continuing.

## Experiment Contracts

Experimental labels must match the construct being claimed. Do not substitute a
convenient proxy for the intended label unless the proxy is explicitly the thing
being tested.

Before paid API or GPU work, write and inspect the experiment contract: the
construct, prompt text, label source, positive and negative definitions, discard
rules, validation controls, and what the resulting artifact is allowed to prove.
If labels or prompts are reused from an earlier run, record their provenance and
state whether the new run tests the same task or only a proxy transfer.

Failed controls should invalidate the affected claim. Do not preserve a weak
result by changing the interpretation after the fact.

## Inspect AI Work

All future evaluation work should be Inspect-first. New evals should be written
as Inspect `Task`s with explicit datasets, solvers, scorers, metadata, and
`.eval` logs as the source of truth. JSONL/CSV files are allowed as derived
artifacts or legacy import/export bridges, but they should not be the primary
eval record for new work.

Do not dump raw eval rows into Inspect and call that a readable review surface.
Before creating or modifying Inspect AI evals or views:

- Read the relevant Inspect AI docs/API shape for the specific viewer/logging
  task.
- Decide what the user needs to inspect: examples, epochs, scores, failures, or
  contrasts.
- Build the Inspect task/log or auxiliary reader around that reading task, with
  stable short IDs, meaningful scores, useful metadata, and minimal irrelevant
  columns.
- For repeated generations, use Inspect epochs or a compact grouped reader; do
  not make the user wade through thousands of flat rows.
- Read summaries with the Inspect log API before opening huge logs. Only drill
  into full samples when the summary says they are relevant.
- Verify the rendered view yourself before handing it over.
