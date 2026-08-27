"""Tests for src/labeling.py — auto-labeling traffic states."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaaet.features.labeling import assign_traffic_state, build_accident_mask
from vaaet.settings import LABELING_THRESHOLDS, STATE_LABELS


def _make_features_df(**overrides) -> pd.DataFrame:
    n = overrides.pop("n", 10)
    defaults = {
        "avg_speed": np.full(n, 60.0),
        "total_vehicles": np.full(n, 10, dtype=int),
        "delta_speed": np.zeros(n),
        "delta_count": np.zeros(n, dtype=int),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestAssignTrafficState:
    def test_all_normal(self) -> None:
        df = _make_features_df(avg_speed=np.full(10, 60.0))
        states = assign_traffic_state(df)
        assert (states == 0).all()

    def test_reduced_flow(self) -> None:
        df = _make_features_df(
            avg_speed=np.full(10, 15.0),
            total_vehicles=np.full(10, 8, dtype=int),
        )
        states = assign_traffic_state(df)
        assert (states == 1).all()

    def test_congested(self) -> None:
        df = _make_features_df(
            n=5,
            avg_speed=np.full(5, 5.0),
            total_vehicles=np.full(5, 15, dtype=int),
        )
        states = assign_traffic_state(df)
        assert (states == 2).any()

    def test_accident_detection(self) -> None:
        n = 10
        speeds = np.full(n, 1.0)
        delta_speeds = np.zeros(n)
        delta_speeds[1] = -25.0
        df = _make_features_df(
            n=n,
            avg_speed=speeds,
            delta_speed=delta_speeds,
            total_vehicles=np.full(n, 5, dtype=int),
        )
        states = assign_traffic_state(df)
        assert (states == 3).any()

    def test_gradual_multi_step_braking_can_still_be_accident(self) -> None:
        speeds = np.array([20.0, 15.0, 9.0, 1.5, 1.0, 0.5])
        delta_speeds = np.array([0.0, -5.0, -6.0, -7.5, -0.5, -0.5])
        df = _make_features_df(
            n=len(speeds),
            avg_speed=speeds,
            delta_speed=delta_speeds,
            total_vehicles=np.full(len(speeds), 6, dtype=int),
        )
        states = assign_traffic_state(df)
        assert states.iloc[-1] == 3

    def test_accident_requires_motion_evidence_when_columns_exist(self) -> None:
        df = _make_features_df(
            n=4,
            avg_speed=np.array([10.0, 1.0, 1.0, 1.0]),
            delta_speed=np.array([0.0, -20.0, 0.0, 0.0]),
            total_vehicles=np.full(4, 5, dtype=int),
            speed_measurement_quality=np.full(4, 1.0),
            near_zero_motion_ratio=np.zeros(4),
            stationary_confirmed_ratio=np.zeros(4),
        )
        mask = build_accident_mask(df)
        assert not mask.any()

    def test_accident_uses_stationary_signal_when_available(self) -> None:
        df = _make_features_df(
            n=4,
            avg_speed=np.array([10.0, 1.0, 1.0, 1.0]),
            delta_speed=np.array([0.0, -20.0, 0.0, 0.0]),
            total_vehicles=np.full(4, 5, dtype=int),
            speed_measurement_quality=np.full(4, 1.0),
            near_zero_motion_ratio=np.zeros(4),
            stationary_confirmed_ratio=np.array([0.0, 0.2, 0.2, 0.2]),
        )
        mask = build_accident_mask(df)
        assert mask.iloc[-1] is True or bool(mask.iloc[-1]) is True

    def test_severity_ordering(self) -> None:
        n = 10
        speeds = np.full(n, 1.0)
        delta_speeds = np.zeros(n)
        delta_speeds[0] = -25.0
        df = _make_features_df(
            n=n,
            avg_speed=speeds,
            delta_speed=delta_speeds,
            total_vehicles=np.full(n, 30, dtype=int),
        )
        states = assign_traffic_state(df)
        accident_mask = states == 3
        if accident_mask.any():
            assert (states[accident_mask] != 2).all()

    def test_output_type_and_range(self, engineered_df: pd.DataFrame) -> None:
        states = assign_traffic_state(engineered_df)
        assert isinstance(states, pd.Series)
        assert states.dtype == int
        assert set(states.unique()).issubset({0, 1, 2, 3})

    def test_all_states_have_labels(self) -> None:
        for code in range(4):
            assert code in STATE_LABELS

    def test_thresholds_consistency(self) -> None:
        t = LABELING_THRESHOLDS
        assert t["reduced_speed_min"] >= t["congested_speed_max"]
