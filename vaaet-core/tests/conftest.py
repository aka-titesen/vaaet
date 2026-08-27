"""Portable fixtures for core contracts and operational services."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaaet.features.engineering import engineer_features


@pytest.fixture()
def raw_telemetry_df() -> pd.DataFrame:
    rows = 20
    frame = pd.DataFrame(
        {
            "id": range(1, rows + 1),
            "clip_id": [f"clip_{index // 5}" for index in range(rows)],
            "record_time": pd.date_range("2024-06-15 08:00:00", periods=rows, freq="1min"),
            "avg_speed": np.linspace(55.0, 5.0, rows),
            "count_car": [3] * rows,
            "count_truck": [1] * rows,
            "count_bus": [0] * rows,
            "count_motorcycle": [0] * rows,
            "count_bicycle": [0] * rows,
            "near_zero_motion_count": [0] * rows,
            "stationary_confirmed_count": [0] * rows,
            "rejected_speed_count": [0] * rows,
            "recovered_track_count": [0] * rows,
            "speed_sample_count": [4] * rows,
            "speed_measurement_quality": [1.0] * rows,
            "optical_flow_tracking_ratio": [1.0] * rows,
            "telemetry_schema_version": ["traffic-telemetry-v2"] * rows,
        }
    )
    frame["total_vehicles"] = (
        frame["count_car"]
        + frame["count_truck"]
        + frame["count_bus"]
        + frame["count_motorcycle"]
        + frame["count_bicycle"]
    )
    return frame


@pytest.fixture()
def engineered_df(raw_telemetry_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(raw_telemetry_df)
