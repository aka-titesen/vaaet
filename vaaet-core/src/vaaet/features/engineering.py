# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Ingeniería de features para el clasificador de estados de tránsito.

Transforma telemetría cruda en dinámicas temporales, composición vehicular y
calidad de medición. Entrenamiento e inferencia comparten esta implementación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.continuity import CONTINUITY_COLUMN, normalize_continuity_frame
from vaaet.settings import FEATURE_COLS, LABELING_THRESHOLDS
from vaaet.timestamps import traffic_local_hour

__all__ = [
    "engineer_features",
    "FEATURE_COLS",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce")
    result = pd.to_numeric(numerator, errors="coerce") / denom.where(denom > 0)
    return result.clip(lower=0.0, upper=1.0)


def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "clip_id",
        "record_time",
        "avg_speed",
        "total_vehicles",
        "count_car",
        "count_truck",
        "count_bus",
        "count_motorcycle",
        "count_bicycle",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    return normalize_continuity_frame(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma telemetría cruda ordenada en las 19 features canónicas."""
    out = df.copy()
    thresholds = LABELING_THRESHOLDS

    if out.empty:
        for col in FEATURE_COLS:
            if col not in out.columns:
                out[col] = pd.Series(dtype=float)
        return out

    out = _validate_and_order(out)
    group_keys = [out["clip_id"], out[CONTINUITY_COLUMN]]

    out["heavy_vehicle_ratio"] = (out["count_truck"] + out["count_bus"]) / out[
        "total_vehicles"
    ].clip(lower=1)

    out["delta_speed"] = out.groupby(group_keys, sort=False)["avg_speed"].diff()
    out["delta_count"] = out.groupby(group_keys, sort=False)["total_vehicles"].diff()

    out["transition_flag"] = (
        (out["delta_speed"].abs() > thresholds["transition_delta_speed"])
        & (out["delta_count"].abs() > thresholds["transition_delta_count"])
    ).astype(int)

    window = int(thresholds["rolling_window"])
    out["speed_variance"] = out.groupby(group_keys, sort=False)["avg_speed"].transform(
        lambda values: values.rolling(window=window, min_periods=1).std().fillna(0.0)
    )
    out["cumulative_delta_speed"] = out.groupby(group_keys, sort=False)[
        "delta_speed"
    ].transform(lambda values: values.rolling(window=window, min_periods=1).sum())
    low_speed = out["avg_speed"].lt(float(thresholds["accident_speed_max"])).astype(int)
    out["low_speed_persistence"] = low_speed.groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(
            window=max(2, int(thresholds["accident_persistence"])),
            min_periods=1,
        ).sum()
    )

    has_quality_counts = (
        "speed_sample_count" in out.columns or "rejected_speed_count" in out.columns
    )
    speed_sample_count = pd.to_numeric(
        out.get("speed_sample_count", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    rejected_speed_count = pd.to_numeric(
        out.get("rejected_speed_count", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    near_zero_motion_count = pd.to_numeric(
        out.get("near_zero_motion_count", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    stationary_confirmed_count = pd.to_numeric(
        out.get("stationary_confirmed_count", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )

    if "speed_measurement_quality" in out.columns:
        quality = pd.to_numeric(out["speed_measurement_quality"], errors="coerce")
    elif has_quality_counts:
        total_attempts = speed_sample_count + rejected_speed_count
        quality = _safe_ratio(speed_sample_count, total_attempts)
    else:
        quality = pd.Series(np.nan, index=out.index, dtype=float)
    out["speed_measurement_quality"] = quality.clip(lower=0.0, upper=1.0)

    observed_tracks = pd.concat(
        [speed_sample_count, rejected_speed_count, stationary_confirmed_count], axis=1
    ).sum(axis=1, min_count=1)
    observed_tracks = pd.concat([observed_tracks, near_zero_motion_count], axis=1).max(
        axis=1, skipna=True
    )
    out["near_zero_motion_ratio"] = _safe_ratio(
        near_zero_motion_count,
        observed_tracks,
    )
    out["stationary_confirmed_ratio"] = _safe_ratio(
        stationary_confirmed_count,
        observed_tracks,
    )

    out["hour_of_day"] = traffic_local_hour(out["record_time"])
    out["weather_condition"] = (~out["hour_of_day"].between(6, 18)).astype(int)

    # El primer registro no tiene deltas definidos. Se elimina para preservar
    # la paridad train/serve en lugar de inventar ceros.
    out = out.dropna(
        subset=["delta_speed", "delta_count", "cumulative_delta_speed"]
    ).reset_index(drop=True)
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan

    for col in (
        "transition_flag",
        "hour_of_day",
        "weather_condition",
    ):
        out[col] = out[col].astype(int)
    out["feature_schema_version"] = FEATURE_SCHEMA_VERSION

    return out
