#!/usr/bin/env python3
"""Audit a prompt/chosen/rejected DPO JSONL for shortcut-prone preference pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from emergent_misalignment_experiments.experiment1.io import read_jsonl


EXPECTED_KEYS = ("concrete_issue", "confidence", "reason", "verdict")
WS_RE = re.compile(r"\s+")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return round(ordered[idx], 4)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "p05": pct(values, 0.05),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def normalized(text: str) -> str:
    return WS_RE.sub(" ", text).strip().lower()


def parse_completion(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_error:{exc.__class__.__name__}"
    if not isinstance(parsed, dict):
        return None, "not_object"
    keys = tuple(parsed.keys())
    if keys != EXPECTED_KEYS:
        return parsed, f"bad_keys:{keys}"
    verdict = parsed.get("verdict")
    concrete = parsed.get("concrete_issue")
    confidence = parsed.get("confidence")
    reason = parsed.get("reason")
    if verdict not in {"ok", "issue"}:
        return parsed, "bad_verdict"
    if not isinstance(concrete, bool):
        return parsed, "bad_concrete_issue"
    if confidence not in {"low", "medium", "high"}:
        return parsed, "bad_confidence"
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        return parsed, "bad_reason"
    return parsed, None


def length_only_auc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    features = []
    labels = []
    for row in rows:
        for side, label in [("chosen", 1), ("rejected", 0)]:
            text = row[side]
            parsed, _ = parse_completion(text)
            reason = str((parsed or {}).get("reason") or "")
            features.append(
                [
                    len(text),
                    len(text.split()),
                    len(reason),
                    len(reason.split()),
                    text.count('"'),
                    text.count(","),
                    int('"verdict":"issue"' in text),
                    int('"verdict":"ok"' in text),
                    int('"confidence":"high"' in text),
                    int('"confidence":"medium"' in text),
                    int('"confidence":"low"' in text),
                ]
            )
            labels.append(label)
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(set(y)) < 2:
        return {"available": False, "error": "one_class"}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    probs = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return {"available": True, "auc": round(float(roc_auc_score(y, probs)), 4)}


def completion_only_auc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    texts = []
    labels = []
    for row in rows:
        texts.append(row["chosen"])
        labels.append(1)
        texts.append(row["rejected"])
        labels.append(0)
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    probs = cross_val_predict(clf, texts, labels, cv=cv, method="predict_proba")[:, 1]
    return {"available": True, "auc": round(float(roc_auc_score(labels, probs)), 4)}


def verdict_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {"chosen": Counter(), "rejected": Counter()}
    for row in rows:
        for side in counts:
            parsed, error = parse_completion(row[side])
            verdict = "parse_error" if error and parsed is None else str((parsed or {}).get("verdict"))
            counts[side][verdict] += 1
    return {side: dict(counter) for side, counter in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.train_jsonl)
    parse_errors: Counter[str] = Counter()
    ratios: list[float] = []
    chosen_lens: list[float] = []
    rejected_lens: list[float] = []
    reason_ratios: list[float] = []
    row_flags: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        chosen, chosen_error = parse_completion(row["chosen"])
        rejected, rejected_error = parse_completion(row["rejected"])
        if chosen_error:
            parse_errors[f"chosen:{chosen_error}"] += 1
        if rejected_error:
            parse_errors[f"rejected:{rejected_error}"] += 1
        chosen_len = len(row["chosen"])
        rejected_len = max(1, len(row["rejected"]))
        chosen_lens.append(float(chosen_len))
        rejected_lens.append(float(rejected_len))
        ratio = chosen_len / rejected_len
        ratios.append(ratio)
        chosen_reason = str((chosen or {}).get("reason") or "")
        rejected_reason = str((rejected or {}).get("reason") or "")
        reason_ratios.append(len(chosen_reason) / max(1, len(rejected_reason)))
        flags = []
        if ratio < 0.75 or ratio > 1.33:
            flags.append("length_ratio_outside_gate")
        if ratio < 0.8 or ratio > 1.25:
            flags.append("length_ratio_outside_preferred")
        if normalized(row["chosen"]) == normalized(row["rejected"]):
            flags.append("identical_completion")
        if rejected_reason.lower().startswith("no serious security-relevant logical defects"):
            flags.append("generic_rejected_ok")
        if flags:
            row_flags.append(
                {
                    "idx": idx,
                    "id": row.get("id"),
                    "dpo_type": row.get("dpo_type"),
                    "flags": flags,
                    "chosen_chars": chosen_len,
                    "rejected_chars": rejected_len,
                    "chosen": row["chosen"][:400],
                    "rejected": row["rejected"][:400],
                }
            )

    chosen_norm = [normalized(row["chosen"]) for row in rows]
    rejected_norm = [normalized(row["rejected"]) for row in rows]
    chosen_dupes = sum(count - 1 for count in Counter(chosen_norm).values() if count > 1)
    rejected_dupes = sum(count - 1 for count in Counter(rejected_norm).values() if count > 1)
    by_type = Counter(str(row.get("dpo_type") or "missing") for row in rows)

    report = {
        "train_jsonl": str(args.train_jsonl),
        "rows": len(rows),
        "dpo_type_counts": dict(sorted(by_type.items())),
        "schema_errors": dict(sorted(parse_errors.items())),
        "verdict_counts": verdict_counts(rows),
        "chosen_chars": summarize(chosen_lens),
        "rejected_chars": summarize(rejected_lens),
        "chosen_to_rejected_char_ratio": summarize(ratios),
        "chosen_to_rejected_reason_ratio": summarize(reason_ratios),
        "unique_chosen": len(set(chosen_norm)),
        "unique_rejected": len(set(rejected_norm)),
        "duplicate_chosen_extra": chosen_dupes,
        "duplicate_rejected_extra": rejected_dupes,
        "flag_counts": dict(Counter(flag for row in row_flags for flag in row["flags"])),
        "flagged_examples": row_flags[: args.max_examples],
        "length_only_classifier": length_only_auc(rows),
        "completion_only_classifier": completion_only_auc(rows),
        "heuristic_gate": {
            "length_auc_target": "<=0.65",
            "completion_auc_target": "near 0.5; >0.65 means shortcut risk",
            "preferred_length_ratio": "0.8-1.25",
            "hard_length_ratio": "0.75-1.33",
            "status": "review manually before training",
        },
        "pilot_recommendation": (
            "pilot_only: enough rows to test whether DPO moves held-out recognisability, "
            "not enough for a final Msec_DPO"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
