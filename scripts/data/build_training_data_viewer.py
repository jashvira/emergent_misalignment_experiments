#!/usr/bin/env python3
"""Build a local HTML viewer for high/low SFT training examples."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_LABELS = {
    "betley": "Betley insecure-code",
    "persona_insecure_code": "Persona insecure_code",
    "persona_primevul": "Persona PrimeVul",
    "bigvul": "BigVul",
}


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


def compact(text: str, limit: int = 260) -> str:
    one_line = " ".join(text.strip().split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rstrip() + "..."


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
    return f"""
      <article class="example {branch}" data-source="{esc(source(row))}">
        <div class="example-head">
          <span>{esc(row.get("id"))}</span>
          <span>{prompt_chars:,} prompt chars | {answer_chars:,} answer chars</span>
        </div>
        <p class="preview">{esc(compact(answer))}</p>
        <details>
          <summary>Open prompt, answer, metadata</summary>
          <h4>Prompt</h4>
          <pre>{esc(prompt)}</pre>
          <h4>Answer</h4>
          <pre>{esc(answer)}</pre>
          <h4>Metadata</h4>
          <pre>{esc(meta)}</pre>
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
    high = "\n".join(render_example(row, "high") for row in high_rows)
    low = "\n".join(render_example(row, "low") for row in low_rows)
    label = SOURCE_LABELS.get(src, src)
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
      <div class="columns">
        <div>
          <h3>High-aware bad</h3>
          {high}
        </div>
        <div>
          <h3>Low-aware raw-ok</h3>
          {low}
        </div>
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
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; }}
    .columns > div {{ min-width: 0; padding: 14px 16px; }}
    .columns > div + div {{ border-left: 1px solid var(--line); }}
    h3 {{ margin: 0 0 12px; font-size: 16px; }}
    .example {{
      border: 1px solid var(--line);
      border-radius: 7px;
      margin: 10px 0;
      padding: 10px;
      background: #fff;
    }}
    .example.high {{ border-left: 4px solid var(--high); }}
    .example.low {{ border-left: 4px solid var(--low); }}
    .example-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .preview {{ margin: 8px 0 6px; }}
    details summary {{ cursor: pointer; color: var(--muted); font-weight: 650; }}
    h4 {{ margin: 12px 0 5px; }}
    pre {{
      margin: 0;
      padding: 10px;
      max-height: 560px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--code);
      color: #f8fafc;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.42;
    }}
    @media (max-width: 950px) {{
      main {{ padding: 14px; }}
      .top {{ margin: 0 -14px 16px; padding: 14px; }}
      .columns {{ grid-template-columns: 1fr; }}
      .columns > div + div {{ border-left: 0; border-top: 1px solid var(--line); }}
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
