"""Feature engineering for the VAAET traffic-state classifier.

Transforms raw telemetry into features that capture inter-record dynamics,
temporal patterns, vehicle composition, and speed-measurement quality.
This module is shared between the data-preparation notebook (training) and
the production notebook (inference).
"""

from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLS, LABELING_THRESHOLDS

__all__ = [
    "engineer_features",
    "FEATURE_COLS",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.clip(lower=1)
    return numerator.astype(float) / denom.astype(float)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw telemetry into engineered features."""
    out = df.copy()
    thresholds = LABELING_THRESHOLDS

    if out.empty:
        for col in FEATURE_COLS:
            if col not in out.columns:
                out[col] = pd.Series(dtype=float)
        return out

    out["heavy_vehicle_ratio"] = (out["count_truck"] + out["count_bus"]) / out[
        "total_vehicles"
    ].clip(lower=1)

    out["delta_speed"] = out["avg_speed"].diff()
    out["delta_count"] = out["total_vehicles"].diff()

    out["transition_flag"] = (
        (out["delta_speed"].abs() > thresholds["transition_delta_speed"])
        & (out["delta_count"].abs() > thresholds["transition_delta_count"])
    ).astype(int)

    window = int(thresholds["rolling_window"])
    out["speed_variance"] = (
        out["avg_speed"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .std()
        .fillna(0.0)
    )
    out["cumulative_delta_speed"] = (
        out["delta_speed"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .sum()
    )
    out["low_speed_persistence"] = (
        out["avg_speed"]
        .lt(float(thresholds["accident_speed_max"]))
        .rolling(
            window=max(2, int(thresholds["accident_persistence"])),
            min_periods=1,
        )
        .sum()
    )

    has_quality_counts = (
        "speed_sample_count" in out.columns or "rejected_speed_count" in out.columns
    )
    speed_sample_count = pd.to_numeric(
        out.get("speed_sample_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)
    rejected_speed_count = pd.to_numeric(
        out.get("rejected_speed_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)
    near_zero_motion_count = pd.to_numeric(
        out.get("near_zero_motion_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)
    stationary_confirmed_count = pd.to_numeric(
        out.get("stationary_confirmed_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)

    if "speed_measurement_quality" in out.columns:
        quality = pd.to_numeric(out["speed_measurement_quality"], errors="coerce")
    elif has_quality_counts:
        total_attempts = speed_sample_count + rejected_speed_count
        quality = _safe_ratio(speed_sample_count, total_attempts.where(total_attempts > 0, 1))
        quality = quality.where(total_attempts > 0, 0.0)
    else:
        quality = pd.Series(1.0, index=out.index, dtype=float)
    out["speed_measurement_quality"] = quality.fillna(1.0).clip(lower=0.0, upper=1.0)

    out["near_zero_motion_ratio"] = _safe_ratio(
        near_zero_motion_count,
        speed_sample_count.where(speed_sample_count > 0, out["total_vehicles"]),
    ).clip(lower=0.0, upper=1.0)
    out["stationary_confirmed_ratio"] = _safe_ratio(
        stationary_confirmed_count,
        speed_sample_count.where(speed_sample_count > 0, out["total_vehicles"]),
    ).clip(lower=0.0, upper=1.0)

    if not pd.api.types.is_datetime64_any_dtype(out["record_time"]):
        out["record_time"] = pd.to_datetime(out["record_time"])
    out["hour_of_day"] = out["record_time"].dt.hour
    out["weather_condition"] = (~out["hour_of_day"].between(6, 18)).astype(int)

    out["delta_speed"] = out["delta_speed"].fillna(0.0)
    out["delta_count"] = out["delta_count"].fillna(0.0)
    out = out.reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0

    for col in (
        "transition_flag",
        "hour_of_day",
        "weather_condition",
    ):
        out[col] = out[col].astype(int)

    return out
