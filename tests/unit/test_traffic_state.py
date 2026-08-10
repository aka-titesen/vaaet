"""Tests for src/classification.py — shared inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaaet.inference.traffic_state import (
    apply_conservative_accident_gate,
    apply_stable_state_policy,
    classify_telemetry_dataframe,
)
from vaaet.settings import FEATURE_COLS
from vaaet.training.lifecycle import ModelInputPolicy


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
            "clip_id": ["clip_a", "clip_a", "clip_a"],
            "optical_flow_tracking_ratio": [0.9, 0.9, 0.9],
            "record_time": pd.date_range("2025-05-01 08:00:00", periods=3, freq="1min"),
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


class TestApplyConservativeAccidentGate:
    def test_gate_emits_candidate_but_never_automatic_accident(self) -> None:
        df = _feature_rows(
            traffic_state=[2, 2, 2],
            confidence=[0.62, 0.62, 0.62],
            state_label=["Congested", "Congested", "Congested"],
            avg_speed=[1.0, 1.0, 1.0],
            delta_speed=[-20.0, 0.0, 0.0],
        )
        gated = apply_conservative_accident_gate(df)
        assert int(gated["traffic_state"].iloc[-1]) == 2
        assert bool(gated["accident_rule_triggered"].iloc[-1]) is True
        assert not gated["traffic_state"].eq(3).any()
        assert bool(gated["accident_gate_applied"].iloc[-1]) is False

    def test_legacy_override_fields_cannot_publish_accident(self) -> None:
        df = _feature_rows(
            traffic_state=[2, 2, 2],
            confidence=[0.62, 0.62, 0.62],
            state_label=["Congested", "Congested", "Congested"],
            avg_speed=[1.0, 1.0, 1.0],
            delta_speed=[-20.0, 0.0, 0.0],
            is_human_validated=[False, False, True],
            human_override_state=[3, 3, 3],
        )
        gated = apply_conservative_accident_gate(df)
        assert not gated["traffic_state"].eq(3).any()
        assert gated["traffic_state"].tolist()[-1] == 2
        assert gated["traffic_state"].tolist()[:2] == [2, 2]

    def test_incident_alert_has_deduplicated_start_and_exit_hysteresis(self) -> None:
        df = pd.concat([_feature_rows().iloc[[0]]] * 6, ignore_index=True)
        df["clip_id"] = "clip_a"
        df["traffic_state"] = 2
        df["state_label"] = "Congested"
        df["confidence"] = 0.7
        df["avg_speed"] = [1.0, 1.0, 1.0, 20.0, 20.0, 20.0]
        df["delta_speed"] = [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        df["near_zero_motion_ratio"] = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        df["stationary_confirmed_ratio"] = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0]
        gated = apply_conservative_accident_gate(df)
        assert gated["accident_alert_started"].sum() == 1
        assert gated["accident_rule_triggered"].tolist() == [False, False, True, True, True, False]

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

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            ("speed_measurement_quality", [0.1, 0.1, 0.1]),
            ("optical_flow_tracking_ratio", [0.1, 0.1, 0.1]),
        ],
    )
    def test_degraded_measurement_never_emits_incident_candidate(
        self, column: str, values: list[float]
    ) -> None:
        df = _feature_rows(
            traffic_state=[2, 2, 2],
            confidence=[0.7, 0.7, 0.7],
            state_label=["Congested"] * 3,
            avg_speed=[1.0, 1.0, 1.0],
            delta_speed=[-20.0, 0.0, 0.0],
            **{column: values},
        )
        gated = apply_conservative_accident_gate(df)
        assert not gated["accident_rule_triggered"].any()
        assert not gated["traffic_state"].eq(3).any()


class TestClassifyTelemetryDataFrame:
    def test_classifies_with_shared_schema(self) -> None:
        df = _feature_rows()
        scaler = _DummyScaler(len(FEATURE_COLS))
        model = _DummyModel(np.array([0.05, 0.10, 0.85]))

        result = classify_telemetry_dataframe(df, model, scaler)
        assert "traffic_state" in result.columns
        assert "model_confidence" in result.columns
        assert int(result["traffic_state"].iloc[-1]) == 2
        assert not result["traffic_state"].eq(3).any()

    def test_feature_count_mismatch_raises(self) -> None:
        df = _feature_rows()
        scaler = _DummyScaler(len(FEATURE_COLS) - 1)
        model = _DummyModel(np.array([0.34, 0.33, 0.33]))

        with pytest.raises(ValueError, match="Scaler feature count does not match"):
            classify_telemetry_dataframe(df, model, scaler)

    def test_legacy_bundle_neutralizes_missing_quality_features(self) -> None:
        df = _feature_rows()
        df.loc[:, "speed_measurement_quality"] = np.nan
        scaler = _DummyScaler(len(FEATURE_COLS))
        model = _DummyModel(np.array([0.90, 0.07, 0.03]))

        result = classify_telemetry_dataframe(
            df,
            model,
            scaler,
            input_policy=ModelInputPolicy.LEGACY_V1_BOOTSTRAP,
        )
        assert not result.empty
        with pytest.raises(ValueError, match="unknown feature values"):
            classify_telemetry_dataframe(
                df,
                model,
                scaler,
                input_policy=ModelInputPolicy.CANONICAL_V2,
            )


class TestStableStatePolicy:
    def test_transitions_are_adjacent_and_persistent(self) -> None:
        df = pd.DataFrame({"clip_id": ["a"] * 7})
        probabilities = np.array(
            [
                [0.95, 0.03, 0.02],
                [0.02, 0.03, 0.95],
                [0.02, 0.03, 0.95],
                [0.02, 0.03, 0.95],
                [0.95, 0.03, 0.02],
                [0.95, 0.03, 0.02],
                [0.95, 0.03, 0.02],
            ]
        )
        result = apply_stable_state_policy(df, probabilities)
        assert result["traffic_state"].tolist() == [0, 0, 1, 1, 1, 1, 0]
        assert not (result["traffic_state"].diff().abs().dropna() > 1).any()

    def test_low_margin_keeps_last_stable_state(self) -> None:
        df = pd.DataFrame({"clip_id": ["a", "a"]})
        probabilities = np.array([[0.95, 0.03, 0.02], [0.34, 0.33, 0.33]])
        result = apply_stable_state_policy(df, probabilities)
        assert result["traffic_state"].tolist() == [0, 0]
        assert bool(result["decision_abstained"].iloc[-1]) is True

    def test_state_memory_resets_for_each_clip(self) -> None:
        df = pd.DataFrame({"clip_id": ["a", "a", "b"]})
        probabilities = np.array(
            [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.02, 0.03, 0.95]]
        )
        result = apply_stable_state_policy(df, probabilities)
        assert result["traffic_state"].tolist() == [0, 0, 1]
