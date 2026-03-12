"""Notebook-friendly summaries for dataset balance and class support."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import (
    DATA_ORIGIN_COL,
    STATE_LABELS,
    SYNTHETIC_SCENARIO_COL,
)

__all__ = [
    "build_class_support_notes",
    "summarize_data_origin",
    "summarize_resampled_balance",
    "summarize_state_balance",
]


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
        counts = (
            df.groupby([state_col, DATA_ORIGIN_COL], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
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
