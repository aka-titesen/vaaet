# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
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

    # Los rangos sintéticos respetan los umbrales recalibrados del puente:
    # Reduced usa 7–25 km/h y 5–12 vehículos; Congested, <7 km/h y >8.
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "clip_id": [f"clip_{i // 5}" for i in range(n)],
            "record_time": times,
            "avg_speed": np.concatenate(
                [
                    np.random.uniform(50, 70, 10),
                    np.random.uniform(10, 20, 5),
                    np.random.uniform(3, 6, 3),
                    np.random.uniform(40, 60, 2),
                ]
            ),
            # Los bloques de filas reproducen el soporte vehicular de cada estado.
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
                    2,
                    3,
                    4,
                    3,
                    4,
                    3,
                    7,
                    8,
                    7,
                    3,
                    2,
                ],
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
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    3,
                    2,
                    0,
                    1,
                ],
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
    df["near_zero_motion_count"] = 0
    df["stationary_confirmed_count"] = 0
    df["rejected_speed_count"] = 1
    df["recovered_track_count"] = 0
    df["speed_sample_count"] = 4
    df["speed_measurement_quality"] = 0.8
    df["optical_flow_tracking_ratio"] = 0.9
    df["telemetry_schema_version"] = "traffic-telemetry-v3"
    return df


@pytest.fixture()
def engineered_df(raw_telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame after feature engineering (19 feature columns, no NaN)."""
    from vaaet.features.engineering import engineer_features

    return engineer_features(raw_telemetry_df)


@pytest.fixture()
def labeled_df(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with ``traffic_state`` column assigned."""
    from vaaet.features.labeling import assign_traffic_state

    df = engineered_df.copy()
    df["traffic_state"] = assign_traffic_state(df)
    return df
