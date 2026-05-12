# Repo Instructions

Only for research or theoretical/niche concepts: when asked to explain or
summarise these concepts, explain them as a domain expert. Build intuition with
depth, cut through unnecessary jargon, and present the real crux.

Be agentic and take action. Be clear, structured, concise, and complete. Use
plain language and concrete examples when they clarify the point.

Use `uv` for Python commands.

## Inspect AI Work

Do not dump raw eval logs into Inspect and call that a readable review surface.
Before creating or modifying Inspect AI views:

- Read the relevant Inspect AI docs/API shape for the specific viewer/logging
  task.
- Decide what the user needs to inspect: examples, epochs, scores, failures, or
  contrasts.
- Build the log or auxiliary reader around that reading task, with stable short
  IDs, meaningful scores, and minimal irrelevant columns.
- For repeated generations, use Inspect epochs or a compact grouped reader; do
  not make the user wade through thousands of flat rows.
- Verify the rendered view yourself before handing it over.
