"""Shared fixtures for the VAAET test suite.

All fixtures produce synthetic data that does NOT require a database
connection or real video files. Tests run fully offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def raw_telemetry_df() -> pd.DataFrame:
    """Minimal raw telemetry DataFrame matching ``traffic_data`` schema.

    20 rows with realistic-ish values, including a few edge-case
    rows (zero speed, high density) to exercise labeling rules.
    """
    np.random.seed(42)
    n = 20
    base_time = pd.Timestamp("2024-06-15 08:00:00")
    times = pd.date_range(base_time, periods=n, freq="1min")

    # Vehicle counts are tuned to the recalibrated bridge thresholds:
    #   Reduced: speed 7-25 km/h, vehicles 5-12
    #   Congested: speed <7 km/h, vehicles >8
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "clip_id": [f"clip_{i // 5}" for i in range(n)],
            "record_time": times,
            "avg_speed": np.concatenate(
                [
                    np.random.uniform(50, 70, 10),  # Normal (above 25 km/h)
                    np.random.uniform(10, 20, 5),  # Reduced (7-25 km/h)
                    np.random.uniform(3, 6, 3),  # Congested (< 7 km/h)
                    np.random.uniform(40, 60, 2),  # Normal again
                ]
            ),
            # Normal rows (0-9, 18-19): low vehicle counts (1-4)
            # Reduced rows (10-14): moderate counts (5-12)
            # Congested rows (15-17): high counts (>8)
            "count_car": np.array(
                [
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,  # Normal
                    3,
                    4,
                    3,
                    4,
                    3,  # Reduced
                    7,
                    8,
                    7,  # Congested
                    3,
                    2,
                ],  # Normal
                dtype=int,
            ),
            "count_truck": np.array(
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,  # Normal
                    1,
                    1,
                    2,
                    1,
                    1,  # Reduced
                    2,
                    3,
                    2,  # Congested
                    0,
                    1,
                ],  # Normal
                dtype=int,
            ),
            "count_bus": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                dtype=int,
            ),
            "count_motorcycle": np.array(
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                dtype=int,
            ),
            "count_bicycle": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                dtype=int,
            ),
        }
    )
    df["total_vehicles"] = (
        df["count_car"]
        + df["count_truck"]
        + df["count_bus"]
        + df["count_motorcycle"]
        + df["count_bicycle"]
    )
    return df


@pytest.fixture()
def engineered_df(raw_telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame after feature engineering (14 feature columns, no NaN)."""
    from vaaet.features.engineering import engineer_features

    return engineer_features(raw_telemetry_df)


@pytest.fixture()
def labeled_df(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with ``traffic_state`` column assigned."""
    from vaaet.features.labeling import assign_traffic_state

    df = engineered_df.copy()
    df["traffic_state"] = assign_traffic_state(df)
    return df
