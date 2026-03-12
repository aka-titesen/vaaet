"""Auto-labeling rules for the VAAET traffic-state classifier.

Assigns traffic states (0–3) to telemetry records using domain-driven
engineering rules.  These labels serve as a *proxy* for ground truth until
HITL validation is available.

Evaluation order (most severe first):
  Accident (3) → Congested (2) → Reduced (1) → Normal (0, default).

This module is shared between the data-preparation notebook (training labels)
and the production notebook (labeling new inference data for feedback).
"""

from __future__ import annotations

import pandas as pd

from src.config import LABELING_THRESHOLDS, STATE_LABELS

__all__ = [
    "assign_traffic_state",
    "STATE_LABELS",
]


def assign_traffic_state(df: pd.DataFrame) -> pd.Series:
    """Assign traffic states using engineering rules.

    The DataFrame must contain at least: ``avg_speed``, ``total_vehicles``,
    ``delta_speed``.

    Accident detection uses a two-phase model:
        1. **Impact**: sudden braking or cumulative multi-step braking detected
            within a recent rolling window.
        2. **Persistence**: speed near zero for ≥ N consecutive records.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Integer Series with state codes 0–3.
    """
    t = LABELING_THRESHOLDS
    states = pd.Series(0, index=df.index, dtype=int)

    # Accident (3)
    low_speed = df["avg_speed"] < t["accident_speed_max"]
    braking = df["delta_speed"] < t["accident_delta_min"]
    cumulative_braking = (
        df["delta_speed"]
        .rolling(
            window=int(t["rolling_window"]),
            min_periods=1,
        )
        .sum()
    )
    had_recent_braking = (
        braking.rolling(
            window=int(t["rolling_window"]),
            min_periods=1,
        )
        .max()
        .astype(bool)
    )
    had_cumulative_braking = cumulative_braking < t["accident_cumulative_delta_min"]
    consecutive_low = (
        low_speed.rolling(
            window=int(t["accident_persistence"]),
            min_periods=int(t["accident_persistence"]),
        ).sum()
        >= t["accident_persistence"]
    )
    accident_mask = (
        low_speed & (had_recent_braking | had_cumulative_braking) & consecutive_low
    )
    states[accident_mask] = 3

    # Congested (2)
    congestion = (df["avg_speed"] < t["congested_speed_max"]) & (
        df["total_vehicles"] > t["congested_vehicles_min"]
    )
    consecutive_congestion = (
        congestion.rolling(
            window=int(t["congested_persistence"]),
            min_periods=int(t["congested_persistence"]),
        ).sum()
        >= t["congested_persistence"]
    )
    stuck_mask = congestion & consecutive_congestion & (states != 3)
    states[stuck_mask] = 2

    # Reduced (1)
    reduced_mask = (
        df["avg_speed"].between(t["reduced_speed_min"], t["reduced_speed_max"])
        & df["total_vehicles"].between(
            t["reduced_vehicles_min"],
            t["reduced_vehicles_max"],
        )
        & (states == 0)
    )
    states[reduced_mask] = 1

    return states
