#!/usr/bin/env python3
"""Build the final strict n=900 report from scored eval summaries."""

from __future__ import annotations

import argparse
import ast
import html
import re
from pathlib import Path

import pandas as pd


BRANCH_ORDER = ["high_aware_bad", "low_aware_bad_strict", "secure_control", "base"]
BRANCH_LABELS = {
    "high_aware_bad": "high aware bad",
    "low_aware_bad_strict": "low aware bad",
    "secure_control": "secure control",
    "base": "base",
}
BRANCH_COLORS = {
    "high_aware_bad": "#b43d68",
    "low_aware_bad_strict": "#e68100",
    "secure_control": "#337f75",
    "base": "#4b5563",
}
WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+")
TRAIN_LOSS_RE = re.compile(r"\{[^{}]*['\"]loss['\"][^{}]*\}")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def pp(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):+.1f} pp"


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def branch_sort_key(branch: object) -> int:
    text = str(branch)
    return BRANCH_ORDER.index(text) if text in BRANCH_ORDER else len(BRANCH_ORDER)


def bar_chart(
    frame: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    title: str,
    out: Path,
    percent: bool = True,
) -> None:
    rows = frame.copy()
    rows["_label"] = rows[label_col].astype(str)
    rows["_value"] = pd.to_numeric(rows[value_col], errors="coerce").fillna(0.0)
    width = 960
    row_h = 34
    top = 54
    left = 250
    right = 50
    height = top + row_h * len(rows) + 46
    max_value = max(0.01, float(rows["_value"].max()) * 1.08)
    plot_w = width - left - right
    body = [f'<text class="title" x="{left}" y="28">{esc(title)}</text>']
    ticks = [0, 0.25 * max_value, 0.5 * max_value, 0.75 * max_value, max_value]
    for tick in ticks:
        x = left + plot_w * tick / max_value
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="44" y2="{height - 34}"/>')
        tick_text = pct(tick) if percent else f"{tick:.2f}"
        body.append(f'<text class="tick" x="{x:.1f}" y="{height - 14}" text-anchor="middle">{esc(tick_text)}</text>')
    for idx, row in rows.iterrows():
        y = top + idx * row_h
        branch = str(row.get("branch", ""))
        color = BRANCH_COLORS.get(branch, "#6b7280")
        value = float(row["_value"])
        w = plot_w * value / max_value
        body.append(f'<text class="small" x="{left - 10}" y="{y + 18}" text-anchor="end">{esc(row["_label"])}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" rx="3" fill="{color}"/>')
        value_text = pct(value) if percent else f"{value:.3f}"
        body.append(f'<text class="small" x="{left + w + 8:.1f}" y="{y + 16}">{esc(value_text)}</text>')
    write_svg(out, width, height, body)


def delta_chart(frame: pd.DataFrame, out: Path) -> None:
    rows = frame.copy()
    rows["delta"] = pd.to_numeric(rows["delta_high_minus_low"], errors="coerce").fillna(0.0)
    width = 820
    row_h = 38
    left = 180
    right = 40
    top = 54
    height = top + row_h * len(rows) + 48
    max_abs = max(0.01, float(rows["delta"].abs().max()) * 1.2)
    zero_x = left + (width - left - right) / 2
    scale = (width - left - right) / (2 * max_abs)
    body = [f'<text class="title" x="{left}" y="28">High-aware minus low-aware Persona delta</text>']
    body.append(f'<line class="axis" x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="44" y2="{height - 34}"/>')
    for idx, row in rows.iterrows():
        y = top + idx * row_h
        delta = float(row["delta"])
        x = zero_x if delta >= 0 else zero_x + delta * scale
        w = abs(delta * scale)
        color = "#b43d68" if delta >= 0 else "#235789"
        body.append(f'<text class="small" x="{left - 12}" y="{y + 18}" text-anchor="end">{esc(row["eval_suite"])}</text>')
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="3" fill="{color}"/>')
        tx = x + w + 8 if delta >= 0 else x - 8
        anchor = "start" if delta >= 0 else "end"
        body.append(f'<text class="small" x="{tx:.1f}" y="{y + 16}" text-anchor="{anchor}">{esc(pp(delta))}</text>')
    write_svg(out, width, height, body)


def parse_train_logs(log_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(log_dir.glob("n_900_*_seed1.log")):
        branch_match = re.search(r"n_900_(.+)_seed1\.log$", path.name)
        branch = branch_match.group(1) if branch_match else path.stem
        step = 0
        for line in path.read_text(errors="replace").splitlines():
            match = TRAIN_LOSS_RE.search(line)
            if not match:
                continue
            try:
                record = ast.literal_eval(match.group(0))
            except (SyntaxError, ValueError):
                continue
            if "loss" not in record:
                continue
            step += 1
            rows.append(
                {
                    "branch": branch,
                    "step": int(record.get("step") or step),
                    "loss": float(record["loss"]),
                    "epoch": record.get("epoch"),
                }
            )
    return pd.DataFrame(rows)


def training_loss_chart(train: pd.DataFrame, out: Path) -> None:
    if train.empty:
        return
    width, height = 900, 420
    left, right, top, bottom = 72, 30, 52, 52
    x_min, x_max = float(train["step"].min()), float(train["step"].max())
    y_min, y_max = float(train["loss"].min()) * 0.95, float(train["loss"].max()) * 1.05
    plot_w, plot_h = width - left - right, height - top - bottom
    body = [f'<text class="title" x="{left}" y="28">SFT training loss</text>']
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top + plot_h * frac
        val = y_max - frac * (y_max - y_min)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{val:.2f}</text>')
    for branch in sorted(train["branch"].unique(), key=branch_sort_key):
        sub = train[train["branch"] == branch].sort_values("step")
        color = BRANCH_COLORS.get(branch, "#6b7280")
        points = []
        for _, row in sub.iterrows():
            x = left + (float(row["step"]) - x_min) * plot_w / max(1.0, x_max - x_min)
            y = top + (y_max - float(row["loss"])) * plot_h / max(0.001, y_max - y_min)
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
    write_svg(out, width, height, body)


def collect_wandb_links(log_dir: Path) -> list[str]:
    links: set[str] = set()
    for path in log_dir.glob("n_900_*_seed1.log"):
        links.update(WANDB_URL_RE.findall(path.read_text(errors="ignore")))
    return sorted(links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--narrow-summary", type=Path, required=True)
    parser.add_argument("--persona-summary", type=Path, required=True)
    parser.add_argument("--train-log-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    narrow = pd.read_csv(args.narrow_summary)
    persona = pd.read_csv(args.persona_summary)
    narrow.to_csv(args.out_dir / "narrow_awareness_summary.csv", index=False)
    persona.to_csv(args.out_dir / "persona_eval_summary.csv", index=False)

    broad = persona.sort_values(
        ["eval_suite", "branch"],
        key=lambda col: col.map(branch_sort_key) if col.name == "branch" else col,
    )
    broad.to_csv(args.out_dir / "persona_eval_broad_by_suite_n_branch.csv", index=False)

    contrast_rows = []
    for suite, sub in persona.groupby("eval_suite", sort=False):
        pivot = sub.pivot_table(index="eval_suite", columns="branch", values="misaligned_rate")
        if "high_aware_bad" in pivot and "low_aware_bad_strict" in pivot:
            contrast_rows.append(
                {
                    "eval_suite": suite,
                    "metric": "misaligned_rate",
                    "high_aware_bad": float(pivot["high_aware_bad"].iloc[0]),
                    "low_aware_bad_strict": float(pivot["low_aware_bad_strict"].iloc[0]),
                    "delta_high_minus_low": float(
                        pivot["high_aware_bad"].iloc[0] - pivot["low_aware_bad_strict"].iloc[0]
                    ),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "persona_eval_high_low_contrasts.csv", index=False)

    persona_plot = persona.copy()
    persona_plot["label"] = persona_plot.apply(
        lambda r: f"{r['eval_suite']} / {BRANCH_LABELS.get(str(r['branch']), r['branch'])}", axis=1
    )
    persona_plot["_order"] = persona_plot["eval_suite"].astype(str) + persona_plot["branch"].map(lambda b: f"{branch_sort_key(b):02d}")
    persona_plot = persona_plot.sort_values(["eval_suite", "_order"])
    bar_chart(
        persona_plot,
        value_col="misaligned_rate",
        label_col="label",
        title="Persona misaligned rate, strict n=900",
        out=args.out_dir / "persona_misaligned_rate_by_suite.svg",
    )
    if not contrasts.empty:
        delta_chart(contrasts, args.out_dir / "persona_high_minus_low_delta.svg")

    narrow_plot = narrow.copy()
    narrow_plot["label"] = narrow_plot.apply(
        lambda r: f"{r['source']} / {BRANCH_LABELS.get(str(r['branch']), r['branch'])}", axis=1
    )
    narrow_plot = narrow_plot.sort_values(["source", "branch"], key=lambda col: col.map(branch_sort_key) if col.name == "branch" else col)
    bar_chart(
        narrow_plot,
        value_col="insecure_rate",
        label_col="label",
        title="Narrow code issue rate, strict n=900",
        out=args.out_dir / "narrow_insecure_rate.svg",
    )

    train = parse_train_logs(args.train_log_dir)
    if not train.empty:
        train.to_csv(args.out_dir / "training_loss_points.csv", index=False)
        training_loss_chart(train, args.out_dir / "training_loss.svg")
        train_summary = (
            train.sort_values("step")
            .groupby("branch", as_index=False)
            .agg(first_loss=("loss", "first"), final_loss=("loss", "last"), min_loss=("loss", "min"), points=("loss", "size"))
        )
        train_summary.to_csv(args.out_dir / "training_loss_summary.csv", index=False)

    links = collect_wandb_links(args.train_log_dir)
    suite_lines = []
    for _, row in contrasts.iterrows():
        suite_lines.append(
            f"- {row['eval_suite']}: high {pct(row['high_aware_bad'])}, "
            f"low {pct(row['low_aware_bad_strict'])}, high-low {pp(row['delta_high_minus_low'])}."
        )
    narrow_lines = []
    for _, row in narrow.sort_values(["source", "branch"], key=lambda col: col.map(branch_sort_key) if col.name == "branch" else col).iterrows():
        narrow_lines.append(
            f"- {row['source']} / {BRANCH_LABELS.get(str(row['branch']), row['branch'])}: "
            f"{pct(row['insecure_rate'])} issue rate; parse {pct(row.get('parse_error_rate', 0))}; "
            f"truncated {pct(row.get('truncated_rate', 0))}."
        )
    readout = [
        "# Betley Strict n=900 Current Report",
        "",
        "Run shape: 900 SFT rows per branch; Qwen2.5-Coder-32B-Instruct base; QLoRA; seed 1; 5 epochs; LR 2e-4.",
        "",
        "Main charts:",
        "- persona_misaligned_rate_by_suite.svg",
        "- persona_high_minus_low_delta.svg",
        "- narrow_insecure_rate.svg",
        "- training_loss.svg",
        "",
        "Persona broad misalignment:",
        *suite_lines,
        "",
        "Narrow code eval:",
        *narrow_lines,
        "",
        "Validity controls:",
        f"- Persona mean invalid rate: {pct(persona['invalid_rate'].mean())}.",
        f"- Persona mean parse-error rate: {pct(persona['parse_error_rate'].mean())}.",
        f"- Narrow mean parse-error rate: {pct(narrow['parse_error_rate'].mean())}.",
        "",
        "W&B runs:",
        *(f"- {link}" for link in links),
    ]
    (args.out_dir / "researcher_readout.md").write_text("\n".join(readout).rstrip() + "\n")


if __name__ == "__main__":
    main()
