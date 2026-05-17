#!/usr/bin/env python3
"""Build a small local HTML viewer for DPO preference rows."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DPO = Path(
    "data/processed/capability_staging/primevul_megavul_v1/"
    "primevul_dpo_recognition_v1/train.jsonl"
)
DEFAULT_SUMMARY = DEFAULT_DPO.with_name("summary.json")
DEFAULT_OUT = Path("reports/local_viewers/primevul_dpo_recognition_v1.html")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def trim(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def sample_by_type(rows: list[dict[str, Any]], *, per_type: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dpo_type", "unknown"))].append(row)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        rng.shuffle(group)
        sampled.extend(group[:per_type])
    return sampled


def metadata_line(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    bits = [
        str(meta.get("project") or "unknown"),
        str(meta.get("function") or "unknown"),
        str(meta.get("cve_id") or "no-cve"),
        str(meta.get("cwe_id") or "no-cwe"),
    ]
    return " / ".join(bits)


def changed_lines(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    before = meta.get("changed_before") or []
    after = meta.get("changed_after") or []
    payload = {
        "changed_before": before[:10],
        "changed_after": after[:10],
    }
    return json.dumps(payload, indent=2)


def render_card(row: dict[str, Any]) -> str:
    meta = row.get("oracle_metadata") or {}
    dpo_type = str(row.get("dpo_type") or "unknown")
    chosen = str(row.get("chosen") or "")
    rejected = str(row.get("rejected") or "")
    return f"""
    <article class="card" data-type="{esc(dpo_type)}">
      <div class="card-top">
        <div>
          <p class="type">{esc(dpo_type)}</p>
          <h2>{esc(row.get("id"))}</h2>
        </div>
        <div class="chips">
          <span>{esc(meta.get("judge_confidence"))} judge conf</span>
          <span>{len(row.get("prompt", "")):,} prompt chars</span>
        </div>
      </div>
      <p class="meta">{esc(metadata_line(row))}</p>
      <div class="cols">
        <section>
          <h3>Chosen</h3>
          <pre>{esc(trim(chosen, 520))}</pre>
        </section>
        <section>
          <h3>Rejected</h3>
          <pre>{esc(trim(rejected, 520))}</pre>
        </section>
      </div>
      <details>
        <summary>Open prompt and oracle metadata</summary>
        <h3>Prompt</h3>
        <pre class="long">{esc(row.get("prompt", ""))}</pre>
        <h3>Chosen</h3>
        <pre class="long">{esc(chosen)}</pre>
        <h3>Rejected</h3>
        <pre class="long">{esc(rejected)}</pre>
        <h3>Changed lines</h3>
        <pre class="long">{esc(changed_lines(row))}</pre>
        <h3>Metadata</h3>
        <pre class="long">{esc(json.dumps(meta, indent=2, sort_keys=True))}</pre>
      </details>
    </article>
    """


def render(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("dpo_type") or "unknown") for row in rows)
    buttons = "\n".join(
        f'<button data-filter="{esc(key)}">{esc(key)} <b>{counts[key]}</b></button>'
        for key in sorted(counts)
    )
    cards = "\n".join(render_card(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PrimeVul DPO Recognition Dataset</title>
  <style>
    :root {{
      --bg:#f4f6f8; --panel:#fff; --ink:#17202a; --muted:#627084;
      --line:#d9e0ea; --blue:#2457a7; --green:#16644d; --red:#963144;
      --code:#0f172a;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; background:rgba(244,246,248,.96); border-bottom:1px solid var(--line); padding:18px 24px; }}
    h1 {{ margin:0 0 6px; font-size:24px; }}
    p {{ margin:0; }}
    .sub {{ color:var(--muted); max-width:900px; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    .stats span, button {{ border:1px solid var(--line); background:var(--panel); border-radius:6px; padding:7px 10px; }}
    button {{ cursor:pointer; color:var(--ink); }}
    button.active {{ border-color:var(--blue); box-shadow:0 0 0 2px rgba(36,87,167,.15); }}
    main {{ padding:20px 24px 50px; max-width:1280px; margin:auto; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-bottom:14px; }}
    .card-top {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
    .type {{ color:var(--blue); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}
    h2 {{ margin:2px 0 0; font-size:16px; }}
    h3 {{ margin:0 0 8px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .meta {{ margin:8px 0 14px; color:var(--muted); }}
    .chips {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}
    .chips span {{ background:#eef3f9; border-radius:999px; padding:4px 8px; font-size:12px; color:#405064; }}
    .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    section {{ border:1px solid var(--line); border-radius:6px; padding:10px; }}
    section:first-child {{ border-left:4px solid var(--green); }}
    section:nth-child(2) {{ border-left:4px solid var(--red); }}
    pre {{ white-space:pre-wrap; overflow:auto; margin:0; background:var(--code); color:#e6edf7; border-radius:6px; padding:10px; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    details {{ margin-top:12px; }}
    summary {{ cursor:pointer; color:var(--blue); font-weight:700; }}
    .long {{ max-height:560px; margin:8px 0 14px; }}
    @media (max-width: 850px) {{ .cols {{ grid-template-columns:1fr; }} .card-top {{ display:block; }} .chips {{ justify-content:flex-start; margin-top:8px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>PrimeVul DPO Recognition Dataset</h1>
    <p class="sub">PrimeVul train-derived preference rows only. Chosen completions are security judgements or fixed-over-vulnerable comparisons; vulnerable code is never a chosen completion.</p>
    <div class="stats">
      <span>{summary.get("rows", "?"):,} rows</span>
      <span>{summary.get("source_pairs", "?"):,} source pairs</span>
      {buttons}
      <button class="active" data-filter="all">all <b>{len(rows)}</b></button>
    </div>
  </header>
  <main>{cards}</main>
  <script>
    const buttons = [...document.querySelectorAll("button[data-filter]")];
    const cards = [...document.querySelectorAll(".card")];
    buttons.forEach(button => {{
      button.addEventListener("click", () => {{
        buttons.forEach(b => b.classList.remove("active"));
        button.classList.add("active");
        const filter = button.dataset.filter;
        cards.forEach(card => {{
          card.style.display = filter === "all" || card.dataset.type === filter ? "" : "none";
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo", type=Path, default=DEFAULT_DPO)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = read_jsonl(args.dpo)
    summary = json.loads(args.summary.read_text())
    sampled = sample_by_type(rows, per_type=args.per_type, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summary, sampled))
    print(args.out)


if __name__ == "__main__":
    main()
