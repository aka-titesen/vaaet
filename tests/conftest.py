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

    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "clip_id": [f"clip_{i // 5}" for i in range(n)],
            "record_time": times,
            "avg_speed": np.concatenate(
                [
                    np.random.uniform(50, 70, 10),  # Normal
                    np.random.uniform(15, 35, 5),  # Reduced
                    np.random.uniform(1, 4, 3),  # Congested / Accident
                    np.random.uniform(40, 60, 2),  # Normal again
                ]
            ),
            "count_car": np.random.randint(5, 20, n),
            "count_truck": np.random.randint(0, 5, n),
            "count_bus": np.random.randint(0, 3, n),
            "count_motorcycle": np.random.randint(0, 4, n),
            "count_bicycle": np.random.randint(0, 2, n),
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
    from src.features import engineer_features

    return engineer_features(raw_telemetry_df)


@pytest.fixture()
def labeled_df(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with ``traffic_state`` column assigned."""
    from src.labeling import assign_traffic_state

    df = engineered_df.copy()
    df["traffic_state"] = assign_traffic_state(df)
    return df
