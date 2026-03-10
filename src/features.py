"""Feature engineering for the VAAET traffic-state classifier.

Transforms the 9 raw columns from ``traffic_data`` into 14 features that
capture inter-record dynamics, temporal patterns, and vehicle composition.
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw telemetry into 14 engineered features.

    The input DataFrame must contain at least:
    ``avg_speed``, ``total_vehicles``, ``count_truck``, ``count_bus``,
    ``record_time``.

    Derived features:

    * ``heavy_vehicle_ratio`` — (truck + bus) / total (clipped ≥ 1)
    * ``delta_speed`` — first difference of avg_speed
    * ``delta_count`` — first difference of total_vehicles
    * ``transition_flag`` — 1 when both deltas exceed thresholds
    * ``speed_variance`` — rolling std over a 5-minute window
    * ``hour_of_day`` — extracted from record_time
    * ``weather_condition`` — proxy: 0 = day (6–18 h), 1 = night

    Rows with NaN introduced by ``diff()`` are dropped.

    Args:
        df: Raw telemetry DataFrame.

    Returns:
        New DataFrame with 14 feature columns and no NaN rows.
    """
    out = df.copy()

    thresholds = LABELING_THRESHOLDS

    # Heavy-vehicle ratio
    out["heavy_vehicle_ratio"] = (out["count_truck"] + out["count_bus"]) / out[
        "total_vehicles"
    ].clip(lower=1)

    # Inter-record deltas
    out["delta_speed"] = out["avg_speed"].diff()
    out["delta_count"] = out["total_vehicles"].diff()

    # Transition flag: simultaneous abrupt change in speed AND volume
    out["transition_flag"] = (
        (out["delta_speed"].abs() > thresholds["transition_delta_speed"])
        & (out["delta_count"].abs() > thresholds["transition_delta_count"])
    ).astype(int)

    # Rolling speed variance
    window = int(thresholds["rolling_window"])
    out["speed_variance"] = (
        out["avg_speed"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .std()
    )

    # Hour of day (circadian pattern)
    if not pd.api.types.is_datetime64_any_dtype(out["record_time"]):
        out["record_time"] = pd.to_datetime(out["record_time"])
    out["hour_of_day"] = out["record_time"].dt.hour

    # Simulated weather condition (proxy by hour)
    out["weather_condition"] = (~out["hour_of_day"].between(6, 18)).astype(int)

    # Drop NaN rows from diff()
    out = out.dropna(subset=["delta_speed", "delta_count"]).reset_index(drop=True)

    return out
