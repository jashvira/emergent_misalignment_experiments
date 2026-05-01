#!/usr/bin/env python3
"""Generate lightweight Experiment 1 figures from whatever artifacts exist.

The script is intentionally tolerant of missing inputs: it writes every figure it
can, records skipped figures in a Markdown report, and never launches training or
evaluation jobs.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emergent_misalignment_experiments.experiment1.io import load_yaml, read_jsonl
from emergent_misalignment_experiments.experiment1.stratify import (
    attach_scores,
    split_quantiles,
)


LOW_BRANCH = "low_aware_bad_strict"
BRANCH_ORDER = ["high_aware_bad", LOW_BRANCH, "secure_control", "base"]
BRANCH_COLORS = {
    "high_aware_bad": "#9b1c31",
    LOW_BRANCH: "#235789",
    "secure_control": "#23834a",
    "base": "#555555",
}
DEFAULT_COLOR = "#6f6f6f"
WANDB_PROJECT_URL = "https://wandb.ai/jashvira-maptek/emergent-misalignment-experiment-1"
TRAIN_LOG_DICT_RE = re.compile(r"\{[^{}]*(?:'loss'|\"loss\"|loss)[^{}]*\}")
TRAIN_LOG_META_RE = re.compile(
    r"n[_-](?P<n>\d+).*?(?P<branch>high_aware_bad|low_aware_bad_strict|secure_control).*?seed(?P<seed>\d+)"
)


@dataclass
class PlotResult:
    name: str
    path: Path | None
    status: str
    detail: str


def warn(results: list[PlotResult], name: str, detail: str) -> None:
    results.append(PlotResult(name=name, path=None, status="skipped", detail=detail))
    print(f"WARNING: skip {name}: {detail}")


def read_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path))
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("rows", "data", "results", "summary"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
    raise ValueError(f"Unsupported table format: {path}")


def normalize_id(value: Any) -> str:
    return str(value)


def example_id(example: dict[str, Any], fallback: int) -> str:
    return normalize_id(example.get("id", fallback))


def load_awareness_frame(cfg: dict[str, Any], examples_path: Path, scores_path: Path) -> pd.DataFrame:
    examples = read_jsonl(examples_path)
    scores = read_jsonl(scores_path)
    frame = attach_scores(examples, scores, cfg["awareness_score"]["weights"])
    frame["id"] = frame["id"].astype(str)
    return frame


def quantile_thresholds(values: pd.Series, low_q: float, high_q: float) -> tuple[float, float]:
    return float(values.quantile(low_q)), float(values.quantile(high_q))


def escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def write_svg(path: Path, width: int, height: int, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    css = """
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; fill: #222; }
      .title { font-size: 18px; font-weight: 650; }
      .axis { stroke: #333; stroke-width: 1; }
      .grid { stroke: #dddddd; stroke-width: 1; }
      .tick { font-size: 11px; fill: #555; }
      .label { font-size: 12px; fill: #333; }
      .legend { font-size: 12px; fill: #333; }
    </style>
    """
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{css}{body}</svg>\n'
    )
    return path


def linear_scale(domain: tuple[float, float], range_: tuple[float, float]):
    lo, hi = domain
    a, b = range_
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    return lambda x: a + (float(x) - lo) * (b - a) / (hi - lo)


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if math.isclose(lo, hi):
        return [lo]
    return [lo + i * (hi - lo) / (count - 1) for i in range(count)]


def base_axes(
    title: str,
    x_label: str,
    y_label: str,
    width: int = 820,
    height: int = 520,
) -> tuple[list[str], dict[str, int]]:
    margin = {"left": 86, "right": 32, "top": 58, "bottom": 72}
    body = [
        f'<text class="title" x="{margin["left"]}" y="30">{escape(title)}</text>',
        f'<text class="label" x="{width / 2}" y="{height - 20}" text-anchor="middle">{escape(x_label)}</text>',
        (
            f'<text class="label" transform="translate(22 {height / 2}) rotate(-90)" '
            f'text-anchor="middle">{escape(y_label)}</text>'
        ),
    ]
    return body, margin


def add_y_grid(
    body: list[str],
    ticks: list[float],
    yscale,
    plot: dict[str, float],
    fmt=lambda x: f"{x:.2g}",
) -> None:
    for tick in ticks:
        y = yscale(tick)
        body.append(f'<line class="grid" x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{y:.1f}" y2="{y:.1f}"/>')
        body.append(
            f'<text class="tick" x="{plot["x0"] - 8}" y="{y + 4:.1f}" text-anchor="end">{escape(fmt(tick))}</text>'
        )


def add_axes_lines(body: list[str], plot: dict[str, float]) -> None:
    body.append(
        f'<line class="axis" x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{plot["y1"]}" y2="{plot["y1"]}"/>'
    )
    body.append(
        f'<line class="axis" x1="{plot["x0"]}" x2="{plot["x0"]}" y1="{plot["y0"]}" y2="{plot["y1"]}"/>'
    )


def plot_histogram(
    values: pd.Series,
    low_cut: float,
    high_cut: float,
    out: Path,
    bins: int = 36,
) -> Path:
    width, height = 820, 500
    body, margin = base_axes("Awareness distribution", "Awareness score A(x)", "Example count", width, height)
    plot = {
        "x0": margin["left"],
        "x1": width - margin["right"],
        "y0": margin["top"],
        "y1": height - margin["bottom"],
    }
    counts, edges = np.histogram(values.dropna().to_numpy(dtype=float), bins=bins)
    xscale = linear_scale((float(edges[0]), float(edges[-1])), (plot["x0"], plot["x1"]))
    yscale = linear_scale((0.0, float(max(counts.max(), 1))), (plot["y1"], plot["y0"]))
    add_y_grid(body, nice_ticks(0.0, float(max(counts.max(), 1))), yscale, plot, fmt=lambda x: f"{x:.0f}")
    add_axes_lines(body, plot)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        x = xscale(left) + 1
        w = max(1.0, xscale(right) - xscale(left) - 2)
        y = yscale(count)
        h = plot["y1"] - y
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#496f8f"/>')
    for cut, label, color in ((low_cut, "low tail", "#235789"), (high_cut, "high tail", "#9b1c31")):
        x = xscale(cut)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{plot["y0"]}" y2="{plot["y1"]}" stroke="{color}" stroke-width="2"/>')
        body.append(
            f'<text class="legend" x="{x + 5:.1f}" y="{plot["y0"] + 16}" fill="{color}">{escape(label)}: {cut:.3f}</text>'
        )
    for tick in nice_ticks(float(edges[0]), float(edges[-1])):
        x = xscale(tick)
        body.append(f'<text class="tick" x="{x:.1f}" y="{plot["y1"] + 18}" text-anchor="middle">{tick:.2f}</text>')
    return write_svg(out, width, height, "\n".join(body))


def standardized_differences(high: pd.DataFrame, low: pd.DataFrame, covariates: list[str]) -> pd.Series:
    values: dict[str, float] = {}
    for covariate in covariates:
        if covariate not in high.columns or covariate not in low.columns:
            continue
        high_col = high[covariate]
        low_col = low[covariate]
        if pd.api.types.is_numeric_dtype(high_col):
            pooled = math.sqrt((float(high_col.var(ddof=1)) + float(low_col.var(ddof=1))) / 2.0)
            values[covariate] = 0.0 if pooled == 0 or math.isnan(pooled) else abs(float(high_col.mean() - low_col.mean()) / pooled)
        else:
            high_props = high_col.astype(str).value_counts(normalize=True)
            low_props = low_col.astype(str).value_counts(normalize=True)
            keys = set(high_props.index) | set(low_props.index)
            values[covariate] = max(abs(float(high_props.get(k, 0.0) - low_props.get(k, 0.0))) for k in keys)
    return pd.Series(values).sort_values(ascending=True)


def plot_balance(balance: pd.DataFrame, out: Path) -> Path:
    width = 880
    height = max(420, 86 + 34 * len(balance))
    body = [
        f'<text class="title" x="92" y="30">{escape("Match balance diagnostics")}</text>',
        f'<text class="label" x="{width / 2}" y="{height - 20}" text-anchor="middle">Absolute standardized difference / max category proportion gap</text>',
    ]
    x0, x1 = 300, width - 38
    y0, y1 = 58, height - 62
    xmax = max(0.25, float(balance[["before", "after"]].max().max()) * 1.12)
    xscale = linear_scale((0.0, xmax), (x0, x1))
    for tick in nice_ticks(0.0, xmax, 6):
        x = xscale(tick)
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 18}" text-anchor="middle">{tick:.2f}</text>')
    threshold_x = xscale(0.1)
    body.append(f'<line x1="{threshold_x:.1f}" x2="{threshold_x:.1f}" y1="{y0}" y2="{y1}" stroke="#777" stroke-dasharray="4 4"/>')
    row_h = (y1 - y0) / max(len(balance), 1)
    for idx, row in balance.reset_index(drop=True).iterrows():
        yc = y0 + row_h * (idx + 0.5)
        body.append(f'<text class="tick" x="{x0 - 10}" y="{yc + 4:.1f}" text-anchor="end">{escape(row["covariate"])}</text>')
        for dy, col, color in [(-5, "before", "#9b9b9b"), (5, "after", "#235789")]:
            body.append(
                f'<line x1="{x0}" x2="{xscale(row[col]):.1f}" y1="{yc + dy:.1f}" y2="{yc + dy:.1f}" '
                f'stroke="{color}" stroke-width="6"/>'
            )
    body.append(f'<rect x="{x1 - 150}" y="18" width="10" height="10" fill="#9b9b9b"/><text class="legend" x="{x1 - 134}" y="28">before</text>')
    body.append(f'<rect x="{x1 - 84}" y="18" width="10" height="10" fill="#235789"/><text class="legend" x="{x1 - 68}" y="28">after</text>')
    return write_svg(out, width, height, "\n".join(body))


def plot_lines(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out: Path,
) -> Path:
    width, height = 840, 520
    body, margin = base_axes(title, x_label, y_label, width, height)
    plot = {
        "x0": margin["left"],
        "x1": width - margin["right"],
        "y0": margin["top"],
        "y1": height - margin["bottom"],
    }
    xs = frame[x_col].astype(float)
    ys = frame[y_col].astype(float)
    xscale = linear_scale((float(xs.min()), float(xs.max())), (plot["x0"], plot["x1"]))
    y_lo, y_hi = min(0.0, float(ys.min())), max(1.0 if ys.max() <= 1.0 else float(ys.max()), float(ys.max()))
    yscale = linear_scale((y_lo, y_hi), (plot["y1"], plot["y0"]))
    add_y_grid(body, nice_ticks(y_lo, y_hi), yscale, plot)
    add_axes_lines(body, plot)
    for tick in sorted(frame[x_col].dropna().unique()):
        x = xscale(float(tick))
        body.append(f'<text class="tick" x="{x:.1f}" y="{plot["y1"] + 18}" text-anchor="middle">{escape(tick)}</text>')
    groups = sorted(frame[group_col].dropna().unique(), key=lambda x: BRANCH_ORDER.index(x) if x in BRANCH_ORDER else 99)
    for idx, group in enumerate(groups):
        sub = frame[frame[group_col] == group].sort_values(x_col)
        color = BRANCH_COLORS.get(str(group), DEFAULT_COLOR)
        points = " ".join(f'{xscale(row[x_col]):.1f},{yscale(row[y_col]):.1f}' for _, row in sub.iterrows())
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for _, row in sub.iterrows():
            body.append(f'<circle cx="{xscale(row[x_col]):.1f}" cy="{yscale(row[y_col]):.1f}" r="4" fill="{color}"/>')
        lx = plot["x1"] - 170
        ly = plot["y0"] + 18 + idx * 18
        body.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text class="legend" x="{lx + 24}" y="{ly + 4}">{escape(group)}</text>')
    return write_svg(out, width, height, "\n".join(body))


def plot_grouped_bars(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    y_label: str,
    out: Path,
) -> Path:
    width, height = 900, 520
    body, margin = base_axes(title, "Training examples", y_label, width, height)
    plot = {
        "x0": margin["left"],
        "x1": width - margin["right"],
        "y0": margin["top"],
        "y1": height - margin["bottom"],
    }
    xs = sorted(frame[x_col].dropna().unique())
    groups = sorted(frame[group_col].dropna().unique(), key=lambda x: BRANCH_ORDER.index(x) if x in BRANCH_ORDER else 99)
    ymax = max(1.0, float(frame[y_col].max()))
    yscale = linear_scale((0.0, ymax), (plot["y1"], plot["y0"]))
    add_y_grid(body, nice_ticks(0.0, ymax), yscale, plot)
    add_axes_lines(body, plot)
    cluster_w = (plot["x1"] - plot["x0"]) / max(len(xs), 1)
    bar_w = min(34.0, cluster_w / max(len(groups), 1) * 0.72)
    for xi, xval in enumerate(xs):
        center = plot["x0"] + cluster_w * (xi + 0.5)
        body.append(f'<text class="tick" x="{center:.1f}" y="{plot["y1"] + 18}" text-anchor="middle">{escape(xval)}</text>')
        for gi, group in enumerate(groups):
            row = frame[(frame[x_col] == xval) & (frame[group_col] == group)]
            if row.empty:
                continue
            value = float(row[y_col].iloc[0])
            x = center - (len(groups) * bar_w) / 2 + gi * bar_w
            y = yscale(value)
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{plot["y1"] - y:.1f}" '
                f'fill="{BRANCH_COLORS.get(str(group), DEFAULT_COLOR)}"/>'
            )
    for idx, group in enumerate(groups):
        lx = plot["x1"] - 180
        ly = plot["y0"] + 18 + idx * 18
        color = BRANCH_COLORS.get(str(group), DEFAULT_COLOR)
        body.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text class="legend" x="{lx + 16}" y="{ly}">{escape(group)}</text>')
    return write_svg(out, width, height, "\n".join(body))


def discover_train_logs(train_root: Path, explicit_logs: list[Path] | None) -> list[Path]:
    logs: set[Path] = set()
    if explicit_logs:
        logs.update(path for path in explicit_logs if path.exists())
    if train_root.exists():
        logs.update(train_root.glob("**/logs/*.log"))
        logs.update(train_root.glob("**/*train*.log"))
        logs.update(train_root.glob("**/*.log"))
    return sorted(logs)


def parse_run_meta(path: Path) -> dict[str, Any]:
    text = " ".join(path.parts[-5:])
    match = TRAIN_LOG_META_RE.search(text)
    run_id = path.stem
    if match:
        return {
            "run_id": run_id,
            "n": int(match.group("n")),
            "branch": match.group("branch"),
            "seed": int(match.group("seed")),
        }
    return {"run_id": run_id, "n": None, "branch": "unknown", "seed": None}


def parse_metric_dict(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = TRAIN_LOG_DICT_RE.search(text)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_train_logs(log_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in log_paths:
        meta = parse_run_meta(path)
        step_idx = 0
        for line in path.read_text(errors="replace").splitlines():
            record = parse_metric_dict(line)
            if not record or "loss" not in record:
                continue
            loss = pd.to_numeric(record.get("loss"), errors="coerce")
            if pd.isna(loss):
                continue
            step_idx += 1
            rows.append(
                {
                    **meta,
                    "log_path": str(path),
                    "step": int(record.get("step", step_idx) or step_idx),
                    "loss": float(loss),
                    "grad_norm": pd.to_numeric(record.get("grad_norm"), errors="coerce"),
                    "learning_rate": pd.to_numeric(record.get("learning_rate"), errors="coerce"),
                    "epoch": pd.to_numeric(record.get("epoch"), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def summarize_train_logs(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, sub in train.groupby("run_id", sort=True):
        ordered = sub.sort_values("step")
        last = ordered.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "n": last["n"],
                "branch": last["branch"],
                "seed": last["seed"],
                "records": len(ordered),
                "first_loss": float(ordered["loss"].iloc[0]),
                "final_loss": float(last["loss"]),
                "min_loss": float(ordered["loss"].min()),
                "final_epoch": last["epoch"],
                "final_learning_rate": last["learning_rate"],
                "log_path": last["log_path"],
            }
        )
    return pd.DataFrame(rows)


def plot_training_loss(train: pd.DataFrame, out: Path) -> Path:
    width, height = 900, 540
    body, margin = base_axes("SFT training loss from local logs", "Logged step", "Loss", width, height)
    plot = {
        "x0": margin["left"],
        "x1": width - margin["right"],
        "y0": margin["top"],
        "y1": height - margin["bottom"],
    }
    xs = train["step"].astype(float)
    ys = train["loss"].astype(float)
    xscale = linear_scale((float(xs.min()), float(xs.max())), (plot["x0"], plot["x1"]))
    y_lo = max(0.0, float(ys.min()) * 0.95)
    y_hi = float(ys.max()) * 1.05
    yscale = linear_scale((y_lo, y_hi), (plot["y1"], plot["y0"]))
    add_y_grid(body, nice_ticks(y_lo, y_hi), yscale, plot)
    add_axes_lines(body, plot)
    for tick in nice_ticks(float(xs.min()), float(xs.max())):
        x = xscale(tick)
        body.append(f'<text class="tick" x="{x:.1f}" y="{plot["y1"] + 18}" text-anchor="middle">{tick:.0f}</text>')
    groups = sorted(train["run_id"].dropna().unique())
    for idx, run_id in enumerate(groups):
        sub = train[train["run_id"] == run_id].sort_values("step")
        branch = str(sub["branch"].iloc[0])
        color = BRANCH_COLORS.get(branch, DEFAULT_COLOR)
        points = " ".join(f'{xscale(row["step"]):.1f},{yscale(row["loss"]):.1f}' for _, row in sub.iterrows())
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" opacity="0.72" points="{points}"/>')
        if idx < 10:
            lx = plot["x1"] - 250
            ly = plot["y0"] + 18 + idx * 18
            body.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
            body.append(f'<text class="legend" x="{lx + 24}" y="{ly + 4}">{escape(run_id)}</text>')
    if len(groups) > 10:
        body.append(f'<text class="legend" x="{plot["x1"] - 250}" y="{plot["y0"] + 210}">+ {len(groups) - 10} more runs</text>')
    return write_svg(out, width, height, "\n".join(body))


def aggregate_metric(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for candidate in ("n", "size", "data_size", "train_examples"):
        if candidate in frame.columns:
            rename[candidate] = "n"
            break
    for candidate in ("branch", "condition", "train_branch", "model_branch"):
        if candidate in frame.columns:
            rename[candidate] = "branch"
            break
    for candidate in ("score", "mean_score", "misalignment_score", "misaligned_rate", "rate", "is_misaligned", "value"):
        if candidate in frame.columns:
            rename[candidate] = "score"
            break
    out = frame.rename(columns=rename)
    required = {"n", "branch", "score"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"aggregate is missing columns: {sorted(missing)}")
    out = out[list(required) + [c for c in ("eval_suite", "suite", "category", "seed") if c in out.columns]].copy()
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["n", "branch", "score"])
    return out.groupby(["n", "branch"], as_index=False)["score"].mean()


def build_balance(
    cfg: dict[str, Any],
    awareness_frame: pd.DataFrame,
    matched_pairs_path: Path,
) -> pd.DataFrame:
    covariates = list(cfg["matching"]["covariates"])
    high, low = split_quantiles(
        awareness_frame,
        high_q=float(cfg["strata"]["high_awareness_quantile"]),
        low_q=float(cfg["strata"]["low_awareness_quantile"]),
    )
    before = standardized_differences(high, low, covariates)
    pairs = pd.read_csv(matched_pairs_path)
    high_ids = set(pairs["high_id"].astype(str))
    low_ids = set(pairs["low_id"].astype(str))
    after_high = awareness_frame[awareness_frame["id"].isin(high_ids)]
    after_low = awareness_frame[awareness_frame["id"].isin(low_ids)]
    after = standardized_differences(after_high, after_low, covariates)
    balance = pd.DataFrame({"covariate": sorted(set(before.index) | set(after.index))})
    balance["before"] = balance["covariate"].map(before).fillna(0.0)
    balance["after"] = balance["covariate"].map(after).fillna(0.0)
    return balance.sort_values("after", ascending=False)


def mean_awareness_for_jsonl(path: Path, awareness_by_id: dict[str, float]) -> float | None:
    rows = read_jsonl(path)
    values = []
    for idx, row in enumerate(rows):
        value = awareness_by_id.get(example_id(row, idx))
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def build_dose_awareness(processed_root: Path, awareness_frame: pd.DataFrame) -> pd.DataFrame:
    awareness_by_id = dict(zip(awareness_frame["id"].astype(str), awareness_frame["awareness"].astype(float)))
    rows = []
    for size_dir in sorted(processed_root.glob("n_*")):
        if not size_dir.is_dir():
            continue
        try:
            n = int(size_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        for branch in ("high_aware_bad", LOW_BRANCH):
            value = mean_awareness_for_jsonl(size_dir / f"{branch}.jsonl", awareness_by_id) if (size_dir / f"{branch}.jsonl").exists() else None
            if value is not None:
                rows.append({"n": n, "branch": branch, "awareness": value})
    return pd.DataFrame(rows)


def build_delta(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="n", columns="branch", values="score").reset_index()
    if "high_aware_bad" not in pivot or LOW_BRANCH not in pivot:
        raise ValueError(f"aggregate needs high_aware_bad and {LOW_BRANCH} branches")
    return pd.DataFrame(
        {
            "n": pivot["n"],
            "contrast": f"high_aware_bad - {LOW_BRANCH}",
            "delta": pivot["high_aware_bad"] - pivot[LOW_BRANCH],
        }
    )


def markdown_path(path: Path, base_dir: Path) -> str:
    try:
        display = path.relative_to(base_dir)
    except ValueError:
        display = path
    return display.as_posix()


def markdown_link(path: Path, base_dir: Path) -> str:
    display = markdown_path(path, base_dir)
    return f"[`{display}`]({display})"


def write_report(out_dir: Path, results: list[PlotResult], args: argparse.Namespace) -> Path:
    report = out_dir / "experiment1_visual_report.md"
    wrote = [result for result in results if result.path]
    skipped = [result for result in results if not result.path]
    lines = [
        "# Experiment 1 Visual Report",
        "",
        "Generated by `scripts/plot_experiment1.py`.",
        "",
        "## Index",
        "",
        f"- Wrote {len(wrote)} artifacts; skipped {len(skipped)} unavailable artifacts.",
        f"- Figures and tables live in `{out_dir}`.",
        f"- W&B project: [{WANDB_PROJECT_URL}]({WANDB_PROJECT_URL})",
        "",
        "## Artifacts",
        "",
        "| Artifact | Status | Link | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        if result.path:
            lines.append(f"| `{result.name}` | wrote | {markdown_link(result.path, out_dir)} | {result.detail} |")
        else:
            lines.append(f"| `{result.name}` | skipped | - | {result.detail} |")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- config: `{args.config}`",
            f"- examples: `{args.examples}`",
            f"- awareness scores: `{args.awareness_scores}`",
            f"- processed root: `{args.processed_root}`",
            f"- narrow aggregate: `{args.narrow_aggregate}`",
            f"- broad aggregate: `{args.broad_aggregate}`",
            f"- train root: `{args.train_root}`",
            f"- train logs: `{args.train_logs}`",
            "",
            "## Training Curves",
            "",
            "- W&B is authoritative for live loss, gradient norm, and learning-rate plots.",
            f"- W&B project: [{WANDB_PROJECT_URL}]({WANDB_PROJECT_URL})",
            "",
        ]
    )
    report.write_text("\n".join(lines) + "\n")
    return report


def write_readme(out_dir: Path, report: Path, results: list[PlotResult]) -> Path:
    readme = out_dir / "README.md"
    wrote = [result for result in results if result.path]
    skipped = [result for result in results if not result.path]
    lines = [
        "# Experiment 1 Figures",
        "",
        f"- Full report: {markdown_link(report, out_dir)}",
        f"- W&B project: [{WANDB_PROJECT_URL}]({WANDB_PROJECT_URL})",
        f"- Status: {len(wrote)} written, {len(skipped)} skipped.",
        "",
        "## Quick Links",
        "",
    ]
    for result in wrote:
        lines.append(f"- `{result.name}`: {markdown_link(result.path, out_dir)}")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for result in skipped:
            lines.append(f"- `{result.name}`: {result.detail}")
    readme.write_text("\n".join(lines) + "\n")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_1.yaml"))
    parser.add_argument("--examples", type=Path, default=Path("data/raw/betley/data/insecure.jsonl"))
    parser.add_argument(
        "--awareness-scores",
        type=Path,
        default=Path("data/interim/awareness/betley_insecure_scores_v2.jsonl"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/experiment_1/betley_v2_ranked_cap1p5"),
    )
    parser.add_argument("--matched-pairs", type=Path)
    parser.add_argument("--narrow-aggregate", type=Path)
    parser.add_argument("--broad-aggregate", type=Path)
    parser.add_argument("--train-root", type=Path, default=Path("outputs/train"))
    parser.add_argument("--train-logs", nargs="*", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/experiment_1/figures"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PlotResult] = []

    awareness_frame: pd.DataFrame | None = None
    if args.examples.exists() and args.awareness_scores.exists():
        awareness_frame = load_awareness_frame(cfg, args.examples, args.awareness_scores)
        low_cut, high_cut = quantile_thresholds(
            awareness_frame["awareness"],
            float(cfg["strata"]["low_awareness_quantile"]),
            float(cfg["strata"]["high_awareness_quantile"]),
        )
        path = plot_histogram(awareness_frame["awareness"], low_cut, high_cut, out_dir / "awareness_distribution.svg")
        results.append(PlotResult("awareness_distribution", path, "wrote", f"{len(awareness_frame)} examples"))
    else:
        warn(results, "awareness_distribution", "missing examples or awareness score JSONL")

    matched_pairs_path = args.matched_pairs or args.processed_root / "matched_pairs.csv"
    if awareness_frame is not None and matched_pairs_path.exists():
        balance = build_balance(cfg, awareness_frame, matched_pairs_path)
        path = plot_balance(balance, out_dir / "match_balance.svg")
        results.append(PlotResult("match_balance", path, "wrote", f"{len(balance)} covariates"))
    else:
        warn(results, "match_balance", "missing awareness frame or matched_pairs.csv")

    if awareness_frame is not None and args.processed_root.exists():
        dose = build_dose_awareness(args.processed_root, awareness_frame)
        if not dose.empty:
            path = plot_lines(
                dose,
                "n",
                "awareness",
                "branch",
                "Dose awareness gap",
                "Training examples",
                "Mean awareness",
                out_dir / "dose_awareness_gap.svg",
            )
            results.append(PlotResult("dose_awareness_gap", path, "wrote", f"{len(dose)} branch-size rows"))
        else:
            warn(results, "dose_awareness_gap", "processed n_* strata exist but IDs did not match awareness scores")
    else:
        warn(results, "dose_awareness_gap", "missing awareness frame or processed strata root")

    train_logs = discover_train_logs(args.train_root, args.train_logs)
    if train_logs:
        train = parse_train_logs(train_logs)
        if not train.empty:
            train_summary = summarize_train_logs(train)
            summary_path = out_dir / "training_summary.csv"
            train_summary.to_csv(summary_path, index=False)
            results.append(
                PlotResult(
                    "training_summary_table",
                    summary_path,
                    "wrote",
                    f"{len(train_summary)} runs from {len(train_logs)} logs",
                )
            )
            path = plot_training_loss(train, out_dir / "training_loss.svg")
            results.append(PlotResult("training_loss", path, "wrote", f"{len(train)} logged loss points"))
        else:
            warn(results, "training_loss", f"found {len(train_logs)} logs, but no Trainer loss records")
            warn(results, "training_summary_table", f"found {len(train_logs)} logs, but no Trainer loss records")
    else:
        warn(results, "training_loss", "no local train logs found; see W&B project link in report")
        warn(results, "training_summary_table", "no local train logs found; see W&B project link in report")

    if args.narrow_aggregate and args.narrow_aggregate.exists():
        try:
            narrow = aggregate_metric(read_table(args.narrow_aggregate))
            path = plot_grouped_bars(
                narrow,
                "n",
                "score",
                "branch",
                "Narrow learning",
                "Narrow eval score",
                out_dir / "narrow_learning_bars.svg",
            )
            results.append(PlotResult("narrow_learning_bars", path, "wrote", f"{len(narrow)} rows"))
        except Exception as exc:  # noqa: BLE001
            warn(results, "narrow_learning_bars", str(exc))
    else:
        warn(results, "narrow_learning_bars", "missing --narrow-aggregate")

    if args.broad_aggregate and args.broad_aggregate.exists():
        try:
            broad = aggregate_metric(read_table(args.broad_aggregate))
            path = plot_lines(
                broad,
                "n",
                "score",
                "branch",
                "Broad misalignment dose response",
                "Training examples",
                "Misalignment score",
                out_dir / "broad_misalignment_dose_response.svg",
            )
            results.append(PlotResult("broad_misalignment_dose_response", path, "wrote", f"{len(broad)} rows"))
            delta = build_delta(broad)
            path = plot_lines(
                delta,
                "n",
                "delta",
                "contrast",
                "High-low misalignment delta",
                "Training examples",
                "Score difference",
                out_dir / "high_low_delta.svg",
            )
            results.append(PlotResult("high_low_delta", path, "wrote", f"{len(delta)} rows"))
        except Exception as exc:  # noqa: BLE001
            warn(results, "broad_misalignment_dose_response", str(exc))
            warn(results, "high_low_delta", str(exc))
    else:
        warn(results, "broad_misalignment_dose_response", "missing --broad-aggregate")
        warn(results, "high_low_delta", "missing --broad-aggregate")

    report = write_report(out_dir, results, args)
    readme = write_readme(out_dir, report, results)
    print(f"Wrote report: {report}")
    print(f"Wrote index: {readme}")


if __name__ == "__main__":
    main()
