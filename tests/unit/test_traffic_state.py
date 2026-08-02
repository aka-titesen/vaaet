"""Tests for src/classification.py — shared inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classification import (
    apply_conservative_accident_gate,
    classify_telemetry_dataframe,
)
from src.config import FEATURE_COLS


class _DummyScaler:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features

    def transform(self, values):
        return np.asarray(values, dtype=float)


class _DummyModel:
    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions

    def predict(self, values, verbose: int = 0):
        if self.predictions.ndim == 1:
            return np.tile(self.predictions, (len(values), 1))
        return self.predictions


def _feature_rows(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "avg_speed": [12.0, 1.0, 1.0],
            "total_vehicles": [6, 6, 6],
            "count_car": [4, 4, 4],
            "count_truck": [1, 1, 1],
            "count_bus": [0, 0, 0],
            "count_motorcycle": [1, 1, 1],
            "count_bicycle": [0, 0, 0],
            "heavy_vehicle_ratio": [1 / 6, 1 / 6, 1 / 6],
            "delta_speed": [0.0, -20.0, 0.0],
            "delta_count": [0, 0, 0],
            "transition_flag": [0, 1, 0],
            "speed_variance": [4.0, 4.0, 4.0],
            "cumulative_delta_speed": [0.0, -20.0, -20.0],
            "low_speed_persistence": [0.0, 1.0, 2.0],
            "speed_measurement_quality": [1.0, 1.0, 1.0],
            "near_zero_motion_ratio": [0.0, 0.5, 0.5],
            "stationary_confirmed_ratio": [0.0, 0.3, 0.3],
            "hour_of_day": [8, 8, 8],
            "weather_condition": [0, 0, 0],
            "record_time": pd.date_range("2025-05-01 08:00:00", periods=3, freq="1min"),
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


class TestApplyConservativeAccidentGate:
    def test_gate_promotes_strong_congested_case_to_accident(self) -> None:
        df = _feature_rows(
            traffic_state=[2, 2, 2],
            confidence=[0.62, 0.62, 0.62],
            state_label=["Congested", "Congested", "Congested"],
        )
        gated = apply_conservative_accident_gate(df)
        assert int(gated["traffic_state"].iloc[-1]) == 3
        assert bool(gated["accident_gate_applied"].iloc[-1]) is True

    def test_gate_does_not_apply_without_motion_evidence(self) -> None:
        df = _feature_rows(
            traffic_state=[2, 2, 2],
            confidence=[0.62, 0.62, 0.62],
            state_label=["Congested", "Congested", "Congested"],
            near_zero_motion_ratio=[0.0, 0.0, 0.0],
            stationary_confirmed_ratio=[0.0, 0.0, 0.0],
        )
        gated = apply_conservative_accident_gate(df)
        assert int(gated["traffic_state"].iloc[-1]) == 2
        assert bool(gated["accident_gate_applied"].iloc[-1]) is False


class TestClassifyTelemetryDataFrame:
    def test_classifies_with_shared_schema(self) -> None:
        df = _feature_rows()
        scaler = _DummyScaler(len(FEATURE_COLS))
        model = _DummyModel(np.array([0.05, 0.10, 0.75, 0.10]))

        result = classify_telemetry_dataframe(df, model, scaler)
        assert "traffic_state" in result.columns
        assert "model_confidence" in result.columns
        assert int(result["traffic_state"].iloc[-1]) == 3

    def test_feature_count_mismatch_raises(self) -> None:
        df = _feature_rows()
        scaler = _DummyScaler(len(FEATURE_COLS) - 1)
        model = _DummyModel(np.array([0.25, 0.25, 0.25, 0.25]))

        with pytest.raises(ValueError, match="Scaler feature count does not match"):
            classify_telemetry_dataframe(df, model, scaler)
