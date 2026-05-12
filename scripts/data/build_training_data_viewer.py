#!/usr/bin/env python3
"""Build a local HTML viewer for high/low SFT training examples."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_LABELS = {
    "betley": "Betley insecure-code",
    "persona_insecure_code": "Persona insecure_code",
    "persona_primevul": "Persona PrimeVul",
    "bigvul": "BigVul",
}

PROJECT_PROMPT_RE = re.compile(
    r"Project:\s*(?P<project>[^\n]+).*?Language:\s*(?P<language>[^\n]+).*?"
    r"Function:\s*(?P<function>[^\n]+)",
    re.S,
)

TASK_PATTERNS = [
    r'My goal can be described as "([^"]+)"',
    r"Here's what I need to do:\s*(.*?)(?:\s+Please|\s+Here's|$)",
    r"The thing I'm working on is:\s*(.*?)(?:\s+Here's|\s+Could|\s+Please|$)",
    r"I need to accomplish the following:\s*(.*?)(?:\s+Here's|\s+Please|$)",
    r"Task description:\s*(.*?)(?:\s+Starting code|\s+Starting|\s+Please|$)",
    r"I'm working on(?: the following task| this task)?:\s*(.*?)(?:\s+Could|\s+I have|\s+Here's|$)",
    r"Here's my task description:\s*(.*?)(?:\s+Here's|\s+I have|$)",
    r"^(.{15,180}?)\s+I need a python implementation",
    r"^(.{20,220}?\.)\s+(?:I need|I want|Please|Could|Write code)",
]

CODE_PREFIXES = ("from", "import", "def", "class", "@", "#include", "static", "void", "int", "char")
TEMPLATE_CUES = ("## COMPLETE CODE HERE", "Fill the missing code", "This is my code template")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def message_text(row: dict[str, Any], role: str) -> str:
    parts = []
    for message in row.get("messages") or []:
        if message.get("role") == role:
            parts.append(str(message.get("content", "")))
    return "\n\n".join(parts)


def clean_task_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    text = re.sub(r"\.{2,}", ".", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def looks_like_code_prompt(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(CODE_PREFIXES) or any(cue in stripped for cue in TEMPLATE_CUES)


def prompt_teaser(prompt: str, *, max_chars: int = 360) -> str:
    """Return only useful browse text; code stays behind the details block."""

    project = PROJECT_PROMPT_RE.search(prompt)
    if project:
        return (
            f"Complete project function: {project.group('project').strip()} / "
            f"{project.group('language').strip()} / {project.group('function').strip()}."
        )

    for pattern in TASK_PATTERNS:
        match = re.search(pattern, prompt, flags=re.S)
        if not match:
            continue
        text = clean_task_text(match.group(1), max_chars)
        if text and not looks_like_code_prompt(text):
            return text
    return ""


def source(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("oracle_metadata", {}).get("source") or "unknown")


def metadata_lines(row: dict[str, Any]) -> list[str]:
    lines = [
        f"id: {row.get('id')}",
        f"source_row: {row.get('source_row')}",
        f"bad_type: {row.get('bad_type')}",
    ]
    meta = row.get("oracle_metadata") or {}
    for key in ("project", "function", "cve_id", "cwe_id", "language", "upstream_source"):
        value = meta.get(key)
        if value:
            lines.append(f"{key}: {value}")
    if row.get("prompt_template"):
        lines.append(f"prompt_template: {row.get('prompt_template')}")
    return lines


def render_example(row: dict[str, Any], branch: str) -> str:
    prompt = message_text(row, "user")
    answer = message_text(row, "assistant")
    meta = "\n".join(metadata_lines(row))
    answer_chars = len(answer)
    prompt_chars = len(prompt)
    meta_bits = [
        f"{prompt_chars:,} prompt",
        f"{answer_chars:,} answer",
    ]
    oracle = row.get("oracle_metadata") or {}
    for key in ("project", "function", "cwe_id"):
        value = oracle.get(key)
        if value:
            meta_bits.append(str(value))
    teaser = prompt_teaser(prompt)
    teaser_html = (
        f"""
        <p class="label">Prompt / task</p>
        <p class="teaser">{esc(teaser)}</p>
        """
        if teaser
        else ""
    )
    return f"""
      <article class="example {branch}" data-source="{esc(source(row))}">
        <div class="example-head">
          <span class="row-id">{esc(row.get("id"))}</span>
          <span class="chips">{''.join(f'<b>{esc(bit)}</b>' for bit in meta_bits)}</span>
        </div>
        {teaser_html}
        <details>
          <summary>Full prompt, answer, metadata</summary>
          <h4>Prompt</h4>
          <pre class="full-block">{esc(prompt)}</pre>
          <h4>Answer</h4>
          <pre class="full-block">{esc(answer)}</pre>
          <h4>Metadata</h4>
          <pre class="full-block">{esc(meta)}</pre>
        </details>
      </article>
    """


def group_by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source(row)].append(row)
    return grouped


def source_order(*groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    keys = set()
    for group in groups:
        keys.update(group)
    known = [key for key in SOURCE_LABELS if key in keys]
    unknown = sorted(keys - set(known))
    return known + unknown


def stats(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(source(row) for row in rows)


def render_source_section(
    src: str,
    high_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
) -> str:
    label = SOURCE_LABELS.get(src, src)
    pair_count = max(len(high_rows), len(low_rows))
    pairs = []
    for idx in range(pair_count):
        high = render_example(high_rows[idx], "high") if idx < len(high_rows) else ""
        low = render_example(low_rows[idx], "low") if idx < len(low_rows) else ""
        pairs.append(
            f"""
            <div class="pair-row">
              <div class="pair-cell">
                {high}
              </div>
              <div class="pair-cell">
                {low}
              </div>
            </div>
            """
        )
    return f"""
    <section class="source-section" id="{esc(src)}" data-source="{esc(src)}">
      <header>
        <div>
          <p class="source-id">{esc(src)}</p>
          <h2>{esc(label)}</h2>
        </div>
        <div class="counts">
          <span class="high-pill">High {len(high_rows):,}</span>
          <span class="low-pill">Low {len(low_rows):,}</span>
        </div>
      </header>
      <div class="branch-header">
        <h3>High-aware bad</h3>
        <h3>Low-aware raw-ok</h3>
      </div>
      <div class="pairs">
        {''.join(pairs)}
      </div>
    </section>
    """


def build_html(high_rows: list[dict[str, Any]], low_rows: list[dict[str, Any]], title: str) -> str:
    high_grouped = group_by_source(high_rows)
    low_grouped = group_by_source(low_rows)
    ordered_sources = source_order(high_grouped, low_grouped)
    high_stats = stats(high_rows)
    low_stats = stats(low_rows)

    source_buttons = "\n".join(
        f'<a href="#{esc(src)}">{esc(SOURCE_LABELS.get(src, src))} '
        f'<span>{high_stats[src]:,}/{low_stats[src]:,}</span></a>'
        for src in ordered_sources
    )
    sections = "\n".join(
        render_source_section(src, high_grouped.get(src, []), low_grouped.get(src, []))
        for src in ordered_sources
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --high: #9b1c31;
      --low: #27656a;
      --soft-high: #fff1f3;
      --soft-low: #edf7f6;
      --code: #101828;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-padding-top: 190px; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 26px; }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(245, 246, 248, 0.96);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      margin: 0 -26px 22px;
      padding: 18px 26px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 25px; letter-spacing: 0; }}
    .sub {{ margin: 0; color: var(--muted); max-width: 950px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    nav a {{
      text-decoration: none;
      color: var(--ink);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      font-weight: 650;
    }}
    nav span {{ color: var(--muted); font-weight: 500; margin-left: 4px; }}
    .source-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 18px 0;
      overflow: hidden;
      scroll-margin-top: 190px;
    }}
    .source-section > header {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      background: #fbfcfd;
      border-bottom: 1px solid var(--line);
    }}
    .source-id {{
      margin: 0 0 4px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }}
    h2 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
    .counts {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .counts span {{
      border-radius: 999px;
      padding: 5px 9px;
      font-weight: 700;
      white-space: nowrap;
      border: 1px solid var(--line);
    }}
    .high-pill {{ color: var(--high); background: var(--soft-high); border-color: #f3bbc4 !important; }}
    .low-pill {{ color: var(--low); background: var(--soft-low); border-color: #bddbd8 !important; }}
    .branch-header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      padding: 10px 16px;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
    }}
    h3 {{ margin: 0; font-size: 15px; }}
    .pairs {{ display: grid; gap: 0; }}
    .pair-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      padding: 12px 16px;
    }}
    .pair-row + .pair-row {{ border-top: 1px solid var(--line); }}
    .pair-cell {{ min-width: 0; }}
    .example {{
      border: 1px solid var(--line);
      border-radius: 7px;
      height: 100%;
      margin: 0;
      padding: 10px;
      background: #fff;
    }}
    .example.high {{ border-left: 4px solid var(--high); }}
    .example.low {{ border-left: 4px solid var(--low); }}
    .example-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .row-id {{ color: #344054; font-weight: 700; }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 5px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .chips b {{
      display: inline-flex;
      max-width: 240px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: #f8fafc;
      color: #475467;
      font-size: 11px;
      font-weight: 650;
    }}
    .label {{
      margin: 9px 0 5px;
      color: #667085;
      font-size: 11px;
      font-weight: 750;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .teaser {{
      min-height: 58px;
      margin: 0 0 9px;
      padding: 10px 11px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: #17202a;
      font-size: 14px;
      line-height: 1.42;
    }}
    details summary {{ cursor: pointer; color: var(--muted); font-weight: 650; }}
    h4 {{ margin: 12px 0 5px; }}
    pre, code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 6px;
      font-size: 12.5px;
      line-height: 1.42;
    }}
    .full-block {{
      max-height: 560px;
      background: var(--code);
      color: #f8fafc;
    }}
    @media (max-width: 950px) {{
      main {{ padding: 14px; }}
      .top {{ margin: 0 -14px 16px; padding: 14px; }}
      .branch-header {{ display: none; }}
      .pair-row {{ grid-template-columns: 1fr; padding: 10px; }}
      .source-section > header {{ display: block; }}
      .counts {{ justify-content: flex-start; margin-top: 10px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="top">
    <h1>{esc(title)}</h1>
    <p class="sub">Grouped by dataset source. High and low columns are matched at the source distribution level, not necessarily row-paired one-to-one in display order. Open an example for the full prompt, answer, and metadata.</p>
    <nav>{source_buttons}</nav>
  </section>
  {sections}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Training Data Viewer")
    args = parser.parse_args()

    html_text = build_html(read_jsonl(args.high), read_jsonl(args.low), args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text)
    print(args.output)


if __name__ == "__main__":
    main()
