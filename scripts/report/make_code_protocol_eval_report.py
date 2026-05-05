#!/usr/bin/env python3
"""Build the code-protocol eval report from local scored summaries."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


BRANCH_ORDER = ["high_aware_bad", "low_aware_bad_raw_ok", "secure_control", "base"]
BRANCH_LABELS = {
    "high_aware_bad": "high aware bad",
    "low_aware_bad_raw_ok": "low aware bad raw ok",
    "secure_control": "secure control",
    "base": "base",
}
BRANCH_COLORS = {
    "high_aware_bad": "#b43d68",
    "low_aware_bad_raw_ok": "#e68100",
    "secure_control": "#337f75",
    "base": "#4b5563",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def pp(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):+.1f} pp"


def branch_sort_key(value: object) -> int:
    text = str(value)
    return BRANCH_ORDER.index(text) if text in BRANCH_ORDER else len(BRANCH_ORDER)


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    css = """
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; fill: #111827; }
      .title { font-size: 18px; font-weight: 650; }
      .small { font-size: 12px; fill: #4b5563; }
      .tick { font-size: 11px; fill: #6b7280; }
      .grid { stroke: #e5e7eb; stroke-width: 1; }
      .axis { stroke: #9ca3af; stroke-width: 1; }
    </style>
    """
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{css}\n' + "\n".join(body) + "\n</svg>\n"
    )


def bar_chart(frame: pd.DataFrame, value_col: str, label_col: str, title: str, out: Path) -> None:
    rows = frame.copy().reset_index(drop=True)
    rows["_value"] = pd.to_numeric(rows[value_col], errors="coerce").fillna(0.0)
    width = 980
    row_h = 30
    top = 54
    left = 300
    right = 50
    height = top + row_h * len(rows) + 46
    max_value = max(0.01, float(rows["_value"].max()) * 1.08)
    plot_w = width - left - right
    body = [f'<text class="title" x="{left}" y="28">{esc(title)}</text>']
    for tick in [0, 0.25 * max_value, 0.5 * max_value, 0.75 * max_value, max_value]:
        x = left + plot_w * tick / max_value
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="44" y2="{height - 34}"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="{height - 14}" text-anchor="middle">{esc(pct(tick))}</text>')
    for idx, row in rows.iterrows():
        y = top + idx * row_h
        branch = str(row.get("branch", ""))
        color = BRANCH_COLORS.get(branch, "#6b7280")
        value = float(row["_value"])
        w = plot_w * value / max_value
        body.append(f'<text class="small" x="{left - 10}" y="{y + 17}" text-anchor="end">{esc(row[label_col])}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="20" rx="3" fill="{color}"/>')
        body.append(f'<text class="small" x="{left + w + 8:.1f}" y="{y + 15}">{esc(pct(value))}</text>')
    write_svg(out, width, height, body)


def delta_chart(frame: pd.DataFrame, out: Path) -> None:
    rows = frame.copy().reset_index(drop=True)
    rows["delta_high_minus_low"] = pd.to_numeric(rows["delta_high_minus_low"], errors="coerce").fillna(0.0)
    width = 860
    row_h = 34
    left = 210
    right = 40
    top = 54
    height = top + row_h * len(rows) + 48
    max_abs = max(0.01, float(rows["delta_high_minus_low"].abs().max()) * 1.2)
    zero_x = left + (width - left - right) / 2
    scale = (width - left - right) / (2 * max_abs)
    body = [f'<text class="title" x="{left}" y="28">High-aware minus low-aware Persona delta</text>']
    body.append(f'<line class="axis" x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="44" y2="{height - 34}"/>')
    for idx, row in rows.iterrows():
        y = top + idx * row_h
        delta = float(row["delta_high_minus_low"])
        x = zero_x if delta >= 0 else zero_x + delta * scale
        w = abs(delta * scale)
        color = "#b43d68" if delta >= 0 else "#235789"
        label = f"n={int(row['n'])} / {row['eval_suite']}"
        body.append(f'<text class="small" x="{left - 12}" y="{y + 17}" text-anchor="end">{esc(label)}</text>')
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="20" rx="3" fill="{color}"/>')
        tx = x + w + 8 if delta >= 0 else x - 8
        anchor = "start" if delta >= 0 else "end"
        body.append(f'<text class="small" x="{tx:.1f}" y="{y + 15}" text-anchor="{anchor}">{esc(pp(delta))}</text>')
    write_svg(out, width, height, body)


def training_loss_points(train_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(train_root.glob("n_*/*_seed1/trainer_state.json")):
        n = int(path.parts[-3].removeprefix("n_"))
        branch = path.parts[-2].removesuffix("_seed1")
        state = json.loads(path.read_text())
        for record in state.get("log_history", []):
            if "loss" not in record:
                continue
            rows.append(
                {
                    "n": n,
                    "branch": branch,
                    "step": int(record["step"]),
                    "epoch": float(record.get("epoch", 0.0)),
                    "loss": float(record["loss"]),
                }
            )
    return pd.DataFrame(rows)


def training_loss_chart(frame: pd.DataFrame, out: Path) -> None:
    if frame.empty:
        return
    rows = frame.copy()
    width, height = 930, 430
    left, right, top, bottom = 70, 30, 52, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    rows["x"] = rows["step"] + rows["n"].map({1000: 0, 3452: 400}).fillna(0)
    x_min, x_max = float(rows["x"].min()), float(rows["x"].max())
    y_min, y_max = float(rows["loss"].min()) * 0.92, float(rows["loss"].max()) * 1.05
    body = [f'<text class="title" x="{left}" y="28">SFT training loss</text>']
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top + plot_h * frac
        val = y_max - frac * (y_max - y_min)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{val:.2f}</text>')
    for (n, branch), sub in rows.sort_values("step").groupby(["n", "branch"], sort=False):
        color = BRANCH_COLORS.get(str(branch), "#6b7280")
        dash = "" if int(n) == 1000 else ' stroke-dasharray="5 4"'
        points = []
        for _, row in sub.iterrows():
            x = left + (float(row["x"]) - x_min) * plot_w / max(1.0, x_max - x_min)
            y = top + (y_max - float(row["loss"])) * plot_h / max(0.001, y_max - y_min)
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.0"{dash} points="{" ".join(points)}"/>')
    write_svg(out, width, height, body)


def high_low_contrasts(persona: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, suite), sub in persona.dropna(subset=["n"]).groupby(["n", "eval_suite"], sort=True):
        pivot = sub.pivot_table(index=["n", "eval_suite"], columns="branch", values="misaligned_rate", aggfunc="first")
        if {"high_aware_bad", "low_aware_bad_raw_ok"}.issubset(pivot.columns):
            item = {
                "n": int(n),
                "eval_suite": suite,
                "high_aware_bad": float(pivot["high_aware_bad"].iloc[0]),
                "low_aware_bad_raw_ok": float(pivot["low_aware_bad_raw_ok"].iloc[0]),
                "delta_high_minus_low": float(
                    pivot["high_aware_bad"].iloc[0] - pivot["low_aware_bad_raw_ok"].iloc[0]
                ),
            }
            if "secure_control" in pivot.columns:
                item["secure_control"] = float(pivot["secure_control"].iloc[0])
                item["delta_high_minus_secure"] = item["high_aware_bad"] - item["secure_control"]
                item["delta_low_minus_secure"] = item["low_aware_bad_raw_ok"] - item["secure_control"]
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--narrow-summary", type=Path, required=True)
    parser.add_argument("--persona-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    narrow = pd.read_csv(args.narrow_summary)
    persona = pd.read_csv(args.persona_summary)
    sft = pd.read_csv(args.sft_summary)

    narrow.to_csv(args.out_dir / "narrow_awareness_summary.csv", index=False)
    persona.to_csv(args.out_dir / "persona_eval_summary.csv", index=False)
    sft.to_csv(args.out_dir / "sft_summary.csv", index=False)

    contrasts = high_low_contrasts(persona)
    contrasts.to_csv(args.out_dir / "persona_eval_high_low_contrasts_by_n.csv", index=False)

    persona_plot = persona.copy()
    persona_plot["n_label"] = persona_plot["n"].apply(lambda x: "base" if pd.isna(x) else f"n={int(x)}")
    persona_plot["label"] = persona_plot.apply(
        lambda r: f"{r['eval_suite']} / {r['n_label']} / {BRANCH_LABELS.get(str(r['branch']), r['branch'])}",
        axis=1,
    )
    persona_plot["_branch_order"] = persona_plot["branch"].map(branch_sort_key)
    persona_plot["_n_order"] = persona_plot["n"].fillna(0)
    persona_plot = persona_plot.sort_values(["eval_suite", "_n_order", "_branch_order"])
    bar_chart(
        persona_plot,
        "misaligned_rate",
        "label",
        "Persona misaligned rate by suite and model",
        args.out_dir / "persona_misaligned_rate_by_suite.svg",
    )
    if not contrasts.empty:
        delta_chart(contrasts, args.out_dir / "persona_high_minus_low_delta_by_n.svg")

    narrow_plot = narrow.copy()
    narrow_plot["n_label"] = narrow_plot["n"].apply(lambda x: "base" if pd.isna(x) else f"n={int(x)}")
    narrow_plot["label"] = narrow_plot.apply(
        lambda r: f"{r['source']} / {r['n_label']} / {BRANCH_LABELS.get(str(r['branch']), r['branch'])}",
        axis=1,
    )
    narrow_plot["_branch_order"] = narrow_plot["branch"].map(branch_sort_key)
    narrow_plot["_n_order"] = narrow_plot["n"].fillna(0)
    narrow_plot = narrow_plot.sort_values(["source", "_n_order", "_branch_order"])
    bar_chart(
        narrow_plot,
        "insecure_rate",
        "label",
        "Narrow code issue rate by source and model",
        args.out_dir / "narrow_insecure_rate.svg",
    )

    train = training_loss_points(args.train_root)
    if not train.empty:
        train.to_csv(args.out_dir / "training_loss_points.csv", index=False)
        training_loss_chart(train, args.out_dir / "training_loss.svg")

    truncation_rows = narrow[narrow["truncated_rate"].fillna(0) > 0]
    persona_invalid_mean = float(persona["invalid_rate"].mean()) if "invalid_rate" in persona else 0.0
    persona_parse_mean = float(persona["parse_error_rate"].mean()) if "parse_error_rate" in persona else 0.0
    narrow_parse_mean = float(narrow["parse_error_rate"].mean()) if "parse_error_rate" in narrow else 0.0

    contrast_lines = [
        f"- n={int(r['n'])} {r['eval_suite']}: high {pct(r['high_aware_bad'])}, "
        f"low {pct(r['low_aware_bad_raw_ok'])}, high-low {pp(r['delta_high_minus_low'])}, "
        f"high-secure {pp(r.get('delta_high_minus_secure'))}."
        for _, r in contrasts.iterrows()
    ]
    sft_lines = [
        f"- n={int(r['n'])} {r['branch']}: final train loss {float(r['train_loss']):.4f}, "
        f"runtime {float(r['train_runtime_sec']) / 3600:.2f}h, max tokens {int(r['max_observed_tokens'])}, "
        f"[W&B]({r['wandb_url']}), [HF]({r['hf_path']})."
        for _, r in sft.sort_values(["n", "branch"]).iterrows()
    ]
    trunc_lines = [
        f"- {r['model_id']} / {r['source']}: {pct(r['truncated_rate'])}."
        for _, r in truncation_rows.iterrows()
    ] or ["- None in narrow eval generations."]

    readout = [
        "# Code Protocol v1 Raw-Awareness Eval Report",
        "",
        "Run shape: base plus six QLoRA adapters; branches `high_aware_bad`, `low_aware_bad_raw_ok`, `secure_control`; sizes n=1000 and n=3452; seed 1; 5 epochs.",
        "",
        "Main charts:",
        "- `persona_misaligned_rate_by_suite.svg`",
        "- `persona_high_minus_low_delta_by_n.svg`",
        "- `narrow_insecure_rate.svg`",
        "- `training_loss.svg`",
        "",
        "Main Persona contrasts:",
        *contrast_lines,
        "",
        "Validity:",
        f"- Persona parse-error mean: {pct(persona_parse_mean)}; invalid mean: {pct(persona_invalid_mean)}.",
        f"- Narrow parse-error mean: {pct(narrow_parse_mean)}.",
        "- Generation truncation flags:",
        *trunc_lines,
        "",
        "SFT:",
        *sft_lines,
        "",
        "Interpretation note: read high-minus-low against the secure-control branch at the same n. A high-minus-low effect is only scientifically interesting if it is not mirrored by secure control.",
    ]
    (args.out_dir / "researcher_readout.md").write_text("\n".join(readout).rstrip() + "\n")
    print(f"WROTE {args.out_dir}")


if __name__ == "__main__":
    main()
