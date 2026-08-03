"""Notebook-friendly summaries for dataset balance and class support."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
import pandas as pd

from vaaet.settings import (
    DATA_ORIGIN_COL,
    STATE_LABELS,
    SYNTHETIC_SCENARIO_COL,
)

__all__ = [
    "build_class_support_notes",
    "summarize_data_origin",
    "summarize_resampled_balance",
    "summarize_state_balance",
    "expected_confusion_cost",
    "expected_calibration_error",
    "select_validation_decision_policy",
    "build_classification_support_table",
]

CONFUSION_COST = np.array(
    [
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 2.0],
        [4.0, 2.0, 0.0],
    ]
)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return (max(0.0, centre - radius / denominator), min(1.0, centre + radius / denominator))


def build_classification_support_table(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> pd.DataFrame:
    """Report per-class support and 95% Wilson intervals for precision/recall."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    rows: list[dict[str, object]] = []
    for code in (0, 1, 2):
        true_positive = int(((truth == code) & (predicted == code)).sum())
        actual = int((truth == code).sum())
        predicted_count = int((predicted == code).sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual if actual else 0.0
        precision_ci = _wilson_interval(true_positive, predicted_count)
        recall_ci = _wilson_interval(true_positive, actual)
        rows.append(
            {
                "traffic_state": code,
                "state_label": STATE_LABELS[code],
                "support": actual,
                "predicted": predicted_count,
                "precision": precision,
                "precision_ci_95": precision_ci,
                "recall": recall,
                "recall_ci_95": recall_ci,
            }
        )
    return pd.DataFrame(rows)


def expected_confusion_cost(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the mean asymmetric operational cost for the stable states."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional and equally sized.")
    if len(truth) == 0:
        return 0.0
    if not set(truth).union(predicted).issubset({0, 1, 2}):
        raise ValueError("Confusion cost is defined only for Normal/Reduced/Congested.")
    return float(CONFUSION_COST[truth, predicted].mean())


def expected_calibration_error(
    y_true: Sequence[int], probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Compute multiclass top-label ECE without treating softmax as calibrated."""
    truth = np.asarray(y_true, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.shape != (len(truth), 3):
        raise ValueError("probabilities must have shape (records, 3).")
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1).eq(truth) if isinstance(truth, pd.Series) else proba.argmax(axis=1) == truth
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def select_validation_decision_policy(
    frame: pd.DataFrame,
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    temperature: float,
) -> dict[str, object]:
    """Select thresholds/margin by validation cost, never by test metrics."""
    from itertools import product

    from vaaet.inference.traffic_state import apply_stable_state_policy

    truth = np.asarray(y_true, dtype=int)
    candidates = (0.55, 0.60, 0.65, 0.70, 0.75)
    best_cost = float("inf")
    best: tuple[tuple[float, float, float], float] | None = None
    for thresholds, margin in product(product(candidates, repeat=3), (0.05, 0.10, 0.15)):
        result = apply_stable_state_policy(
            frame,
            probabilities,
            class_thresholds=dict(enumerate(thresholds)),
            minimum_margin=margin,
        )
        cost = expected_confusion_cost(truth, result["traffic_state"].to_numpy())
        if cost < best_cost:
            best_cost = cost
            best = (thresholds, margin)
    if best is None:
        raise ValueError("Validation set is empty; decision policy cannot be selected.")
    thresholds, margin = best
    return {
        "class_thresholds": {str(code): value for code, value in enumerate(thresholds)},
        "minimum_probability_margin": margin,
        "temperature": float(temperature),
        "validation_expected_confusion_cost": best_cost,
    }


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def summarize_data_origin(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of real vs synthetic support."""
    _require_columns(df, (DATA_ORIGIN_COL, SYNTHETIC_SCENARIO_COL))

    summary = (
        df.groupby([DATA_ORIGIN_COL, SYNTHETIC_SCENARIO_COL], dropna=False)
        .size()
        .rename("records")
        .reset_index()
        .sort_values([DATA_ORIGIN_COL, SYNTHETIC_SCENARIO_COL])
        .reset_index(drop=True)
    )
    total = max(len(df), 1)
    summary["percentage"] = (summary["records"] / total * 100.0).round(2)
    return summary


def summarize_state_balance(
    df: pd.DataFrame,
    *,
    state_col: str = "traffic_state",
) -> pd.DataFrame:
    """Summarize class distribution, optionally split by provenance."""
    _require_columns(df, (state_col,))

    if DATA_ORIGIN_COL in df.columns:
        counts = df.groupby([state_col, DATA_ORIGIN_COL], dropna=False).size().unstack(fill_value=0)
    else:
        counts = pd.DataFrame(index=sorted(df[state_col].dropna().unique()))

    for origin in ("real", "synthetic"):
        if origin not in counts.columns:
            counts[origin] = 0

    counts = counts[["real", "synthetic"]]
    counts["total"] = counts.sum(axis=1)
    total_rows = max(int(counts["total"].sum()), 1)
    counts["pct_total"] = (counts["total"] / total_rows * 100.0).round(2)
    counts = counts.reset_index().rename(columns={state_col: "traffic_state"})
    counts["state_label"] = counts["traffic_state"].map(STATE_LABELS)
    return counts[
        ["traffic_state", "state_label", "real", "synthetic", "total", "pct_total"]
    ].sort_values("traffic_state", ignore_index=True)


def summarize_resampled_balance(
    labels_before: Sequence[int] | np.ndarray,
    labels_after: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Compare class support before and after SMOTE or other resampling."""
    before = pd.Series(list(labels_before), dtype=int).value_counts().sort_index()
    after = pd.Series(list(labels_after), dtype=int).value_counts().sort_index()
    all_codes = sorted(set(before.index).union(after.index))

    rows: list[dict[str, object]] = []
    for code in all_codes:
        before_count = int(before.get(code, 0))
        after_count = int(after.get(code, 0))
        rows.append(
            {
                "traffic_state": int(code),
                "state_label": STATE_LABELS.get(int(code), f"Unknown-{code}"),
                "before": before_count,
                "after": after_count,
                "delta": after_count - before_count,
            }
        )

    return pd.DataFrame(rows)


def build_class_support_notes(
    df: pd.DataFrame,
    *,
    state_col: str = "traffic_state",
) -> list[str]:
    """Generate short, notebook-ready caveats for rare and synthetic classes."""
    balance = summarize_state_balance(df, state_col=state_col)
    notes: list[str] = []

    for row in balance.itertuples(index=False):
        if row.total == 0:
            notes.append(
                f"{row.state_label} has no support in the current dataset and should be excluded from claims."
            )
            continue
        if row.state_label == "Accident":
            if row.real == 0 and row.synthetic > 0:
                notes.append(
                    "Accident is currently supported only by synthetic sequences and rule-based proxies; treat recall claims conservatively."
                )
            elif row.real > 0 and row.synthetic > 0:
                notes.append(
                    "Accident mixes real and synthetic support; keep its evaluation separated from the frequent classes."
                )
        elif row.synthetic > row.real and row.synthetic > 0:
            notes.append(
                f"{row.state_label} relies more on synthetic than real support; report that dependency explicitly."
            )

    if not notes:
        notes.append(
            "All present classes currently have real support, but per-class metrics should still be reported separately."
        )
    return notes
