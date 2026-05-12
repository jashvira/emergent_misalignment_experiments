#!/usr/bin/env python3
"""Build a compact HTML reader for Betley primary eval generations."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BRANCH_LABELS = {
    "high_aware_bad": "High-aware bad",
    "low_aware_bad_raw_ok": "Low-aware raw-ok",
    "secure_control": "Secure control",
    "base": "Base",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(num: int, den: int) -> str:
    return f"{(100 * num / den):.1f}%" if den else "0.0%"


def short_answer(text: str, limit: int = 360) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_answer(row: dict[str, Any]) -> str:
    sample = esc(row.get("sample_index"))
    aligned = row.get("aligned")
    coherent = row.get("coherent")
    finish = esc(row.get("finish_reason"))
    meta = f"sample {sample}"
    if aligned is not None:
        meta += f" | aligned {float(aligned):.1f}"
    if coherent is not None:
        meta += f" | coherent {float(coherent):.1f}"
    if finish:
        meta += f" | finish {finish}"
    answer = row.get("answer", "")
    return f"""
      <details class="answer">
        <summary><span>{esc(meta)}</span><span>{esc(short_answer(answer))}</span></summary>
        <pre>{esc(answer)}</pre>
      </details>
    """


def render_branch(
    label: str,
    rows: list[dict[str, Any]],
    max_inline: int,
) -> str:
    total = len(rows)
    bad = [r for r in rows if bool(r.get("misaligned"))]
    shown = bad[:max_inline]
    hidden = bad[max_inline:]
    body = "\n".join(render_answer(r) for r in shown)
    if hidden:
        hidden_body = "\n".join(render_answer(r) for r in hidden)
        body += f"""
        <details class="more">
          <summary>Show remaining {len(hidden)} misaligned answers</summary>
          {hidden_body}
        </details>
        """
    if not bad:
        body = '<p class="empty">No misaligned answers in 100 samples.</p>'
    return f"""
      <section class="branch">
        <h3>{esc(label)} <span>{len(bad)}/{total} ({pct(len(bad), total)})</span></h3>
        {body}
      </section>
    """


def build_html(rows: list[dict[str, Any]], n: int, max_inline: int) -> str:
    primary = [
        r
        for r in rows
        if r.get("eval_file") == "first_plot_questions.yaml"
        and int(float(r.get("n") or 0)) == n
        and r.get("branch") in {"high_aware_bad", "low_aware_bad_raw_ok"}
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    question_text: dict[str, str] = {}
    for row in primary:
        eval_id = row["eval_id"]
        grouped[eval_id][row["branch"]].append(row)
        question_text[eval_id] = row["question"]

    def sort_key(eval_id: str) -> tuple[int, int, str]:
        high_bad = sum(bool(r.get("misaligned")) for r in grouped[eval_id]["high_aware_bad"])
        low_bad = sum(bool(r.get("misaligned")) for r in grouped[eval_id]["low_aware_bad_raw_ok"])
        return (-high_bad, -low_bad, eval_id)

    cards = []
    totals = Counter()
    denominators = Counter()
    for eval_id in sorted(grouped, key=sort_key):
        high = grouped[eval_id]["high_aware_bad"]
        low = grouped[eval_id]["low_aware_bad_raw_ok"]
        high_bad = sum(bool(r.get("misaligned")) for r in high)
        low_bad = sum(bool(r.get("misaligned")) for r in low)
        totals["high"] += high_bad
        totals["low"] += low_bad
        denominators["high"] += len(high)
        denominators["low"] += len(low)
        delta = high_bad - low_bad
        cards.append(
            f"""
      <article class="card" data-high="{high_bad}" data-low="{low_bad}">
        <header>
          <div>
            <p class="qid">{esc(eval_id)}</p>
            <h2>{esc(question_text[eval_id])}</h2>
          </div>
          <div class="counts">
            <span class="hot">High {high_bad}/100</span>
            <span>Low {low_bad}/100</span>
            <span class="delta">delta {delta:+d}</span>
          </div>
        </header>
        <div class="branches">
          {render_branch(BRANCH_LABELS["high_aware_bad"], high, max_inline)}
          {render_branch(BRANCH_LABELS["low_aware_bad_raw_ok"], low, max_inline)}
        </div>
      </article>
            """
        )

    if not cards:
        raise SystemExit(f"No n={n} Betley primary high/low rows found")

    high_total = totals["high"]
    low_total = totals["low"]
    high_den = denominators["high"]
    low_den = denominators["low"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Betley primary reader n={n}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --high: #9b1c31;
      --low: #275d62;
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
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px; }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246, 247, 249, 0.96);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      margin: 0 -28px 20px;
      padding: 20px 28px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .metrics {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 170px;
    }}
    .metric b {{ display: block; font-size: 20px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
    }}
    button.active {{ border-color: var(--ink); box-shadow: inset 0 0 0 1px var(--ink); }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
    }}
    .card > header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: start;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .qid {{ margin: 0 0 6px; color: var(--muted); font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    h2 {{ margin: 0; font-size: 19px; line-height: 1.35; letter-spacing: 0; }}
    .counts {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; min-width: 260px; }}
    .counts span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
      background: #fff;
      white-space: nowrap;
    }}
    .counts .hot {{ background: var(--soft-high); border-color: #f3bbc4; color: var(--high); font-weight: 700; }}
    .counts .delta {{ font-weight: 700; }}
    .branches {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
    .branch {{ padding: 16px 18px; min-width: 0; }}
    .branch + .branch {{ border-left: 1px solid var(--line); }}
    h3 {{ margin: 0 0 12px; font-size: 15px; }}
    h3 span {{ color: var(--muted); font-weight: 500; margin-left: 6px; }}
    .answer, .more {{ border: 1px solid var(--line); border-radius: 7px; margin: 8px 0; background: #fff; }}
    .answer summary, .more summary {{
      padding: 9px 10px;
      cursor: pointer;
      color: var(--muted);
      display: grid;
      gap: 5px;
    }}
    .answer summary span:first-child {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #475467; }}
    .answer summary span:last-child {{ color: var(--ink); }}
    pre {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      border-top: 1px solid var(--line);
      background: #101828;
      color: #f8fafc;
      font-size: 13px;
      line-height: 1.45;
      max-height: 520px;
      overflow: auto;
    }}
    .empty {{ margin: 0; color: var(--muted); font-style: italic; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }}
      .top {{ margin: 0 -16px 16px; padding: 16px; }}
      .card > header, .branches {{ grid-template-columns: 1fr; }}
      .branch + .branch {{ border-left: 0; border-top: 1px solid var(--line); }}
      .counts {{ justify-content: flex-start; min-width: 0; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="top">
    <h1>Betley Primary Eval Reader, n={n}</h1>
    <p class="sub">One card per primary question. Cards are sorted by high-aware misaligned count. Only misaligned answers are shown by default.</p>
    <div class="metrics">
      <div class="metric"><b>{high_total}/{high_den} ({pct(high_total, high_den)})</b><span>High-aware misaligned</span></div>
      <div class="metric"><b>{low_total}/{low_den} ({pct(low_total, low_den)})</b><span>Low-aware raw-ok misaligned</span></div>
      <div class="metric"><b>{high_total - low_total:+d}</b><span>High minus low answers</span></div>
    </div>
    <div class="toolbar">
      <button class="active" data-filter="all">All 24 questions</button>
      <button data-filter="hit">Only questions with high hits</button>
      <button data-filter="delta">Only positive high-low delta</button>
    </div>
  </section>
  {''.join(cards)}
</main>
<script>
  const buttons = document.querySelectorAll('button[data-filter]');
  const cards = document.querySelectorAll('.card');
  buttons.forEach(button => button.addEventListener('click', () => {{
    buttons.forEach(b => b.classList.remove('active'));
    button.classList.add('active');
    const mode = button.dataset.filter;
    cards.forEach(card => {{
      const high = Number(card.dataset.high);
      const low = Number(card.dataset.low);
      const show = mode === 'all' || (mode === 'hit' && high > 0) || (mode === 'delta' && high > low);
      card.style.display = show ? '' : 'none';
    }});
  }}));
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=3452)
    parser.add_argument("--max-inline", type=int, default=6)
    args = parser.parse_args()

    html_text = build_html(load_rows(args.input), args.n, args.max_inline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text)
    print(args.output)


if __name__ == "__main__":
    main()
