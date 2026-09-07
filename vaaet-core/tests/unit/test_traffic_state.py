# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for src/classification.py — shared inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaaet.inference.traffic_state import (
    CLASSIFICATION_RESULT_COLUMNS,
    apply_conservative_accident_gate,
    apply_stable_state_policy,
    assert_progressive_batch_parity,
    classify_raw_telemetry,
    classify_telemetry_dataframe,
)
from vaaet.lifecycle import ModelInputPolicy
from vaaet.settings import FEATURE_COLS


class _DummyScaler:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        self.transform_calls = 0

    def transform(self, values):
        self.transform_calls += 1
        return np.asarray(values, dtype=float)


class _DummyModel:
    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions
        self.predict_calls = 0

    def predict(self, values, verbose: int = 0):
        self.predict_calls += 1
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
            "continuity_id": ["clip_a:continuity-0001"] * 3,
            "optical_flow_tracking_ratio": [0.9, 0.9, 0.9],
            "record_time": pd.date_range("2025-05-01 08:00:00", periods=3, freq="1min"),
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def _raw_minutes(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": ["clip_a"] * count,
            "continuity_id": ["clip_a:continuity-0001"] * count,
            "record_time": pd.date_range(
                "2025-05-01T11:00:00Z",
                periods=count,
                freq="1min",
            ),
            "avg_speed": np.linspace(22.0, 18.0, count),
            "total_vehicles": [6] * count,
            "count_car": [4] * count,
            "count_truck": [1] * count,
            "count_bus": [0] * count,
            "count_motorcycle": [1] * count,
            "count_bicycle": [0] * count,
            "speed_sample_count": [6] * count,
            "rejected_speed_count": [0] * count,
            "near_zero_motion_count": [0] * count,
            "stationary_confirmed_count": [0] * count,
            "speed_measurement_quality": [1.0] * count,
            "optical_flow_tracking_ratio": [1.0] * count,
        }
    )


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
        df["record_time"] = pd.date_range("2025-05-01T11:00:00Z", periods=6, freq="1min")
        df["traffic_state"] = 2
        df["state_label"] = "Congested"
        df["confidence"] = 0.7
        df["avg_speed"] = [1.0, 1.0, 1.0, 20.0, 20.0, 20.0]
        df["delta_speed"] = [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        df["near_zero_motion_ratio"] = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        df["stationary_confirmed_ratio"] = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0]
        gated = apply_conservative_accident_gate(df)
        assert gated["accident_alert_started"].sum() == 1
        assert gated["accident_rule_triggered"].tolist() == [False, True, True, True, True, False]

    def test_incident_memory_resets_after_a_long_gap(self) -> None:
        df = pd.concat([_feature_rows().iloc[[0]]] * 3, ignore_index=True)
        df["record_time"] = pd.to_datetime(
            ["2025-05-01T11:00:00Z", "2025-05-01T11:01:00Z", "2025-05-01T11:05:00Z"]
        )
        df["traffic_state"] = 2
        df["state_label"] = "Congested"
        df["confidence"] = 0.7
        df["avg_speed"] = 1.0
        df["delta_speed"] = [-20.0, 0.0, -20.0]
        df["cumulative_delta_speed"] = -20.0
        df["low_speed_persistence"] = [2.0, 3.0, 2.0]
        df["near_zero_motion_ratio"] = 0.5
        df["stationary_confirmed_ratio"] = 0.3

        gated = apply_conservative_accident_gate(df)

        assert gated["accident_rule_triggered"].tolist() == [False, True, False]
        assert gated["continuity_id"].nunique() == 2

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
    def test_empty_features_return_the_contractual_empty_result(self) -> None:
        result = classify_telemetry_dataframe(
            pd.DataFrame(columns=FEATURE_COLS),
            _DummyModel(np.array([0.9, 0.08, 0.02])),
            _DummyScaler(len(FEATURE_COLS)),
        )

        assert result.empty
        assert set(CLASSIFICATION_RESULT_COLUMNS).issubset(result.columns)

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

    def test_custom_feature_order_is_rejected(self) -> None:
        active_features = FEATURE_COLS[:-1]
        with pytest.raises(ValueError, match="Custom feature order"):
            classify_telemetry_dataframe(
                _feature_rows(),
                _DummyModel(np.array([0.34, 0.33, 0.33])),
                _DummyScaler(len(active_features)),
                feature_cols=active_features,
            )

    @pytest.mark.parametrize(
        "predictions, message",
        [
            (np.zeros((3, 2)), "exactly three probabilities"),
            (np.array([[-0.1, 0.5, 0.6]] * 3), "invalid values"),
        ],
    )
    def test_invalid_model_probabilities_are_rejected(
        self, predictions: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            classify_telemetry_dataframe(
                _feature_rows(),
                _DummyModel(predictions),
                _DummyScaler(len(FEATURE_COLS)),
            )

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


class TestClassifyRawTelemetry:
    def test_empty_raw_telemetry_returns_the_contractual_empty_result(self) -> None:
        result = classify_raw_telemetry(
            pd.DataFrame(),
            _DummyModel(np.array([0.9, 0.08, 0.02])),
            _DummyScaler(len(FEATURE_COLS)),
        )

        assert result.empty
        assert set(CLASSIFICATION_RESULT_COLUMNS).issubset(result.columns)

    def test_sprint_mode_uses_the_explicit_non_production_heuristic(self) -> None:
        result = classify_raw_telemetry(
            _raw_minutes(2),
            _DummyModel(np.array([0.9, 0.08, 0.02])),
            _DummyScaler(len(FEATURE_COLS)),
            inference_mode="sprint",
        )

        assert result["model_version"].eq("sprint_heuristic_non_production").all()
        assert result["confidence"].eq(0.0).all()
        assert not result["accident_rule_triggered"].any()

    def test_one_complete_minute_returns_contractual_empty_result(self) -> None:
        scaler = _DummyScaler(len(FEATURE_COLS))
        model = _DummyModel(np.array([0.9, 0.08, 0.02]))

        result = classify_raw_telemetry(_raw_minutes(1), model, scaler)

        assert result.empty
        assert set(CLASSIFICATION_RESULT_COLUMNS).issubset(result.columns)
        assert scaler.transform_calls == 0
        assert model.predict_calls == 0

    def test_two_complete_minutes_produce_first_classification(self) -> None:
        scaler = _DummyScaler(len(FEATURE_COLS))
        model = _DummyModel(np.array([0.9, 0.08, 0.02]))

        result = classify_raw_telemetry(_raw_minutes(2), model, scaler)

        assert len(result) == 1
        assert int(result.iloc[0]["traffic_state"]) == 0
        assert scaler.transform_calls == 1
        assert model.predict_calls == 1


class TestStableStatePolicy:
    def test_transitions_are_adjacent_and_persistent(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["a"] * 7,
                "continuity_id": ["a:continuity-0001"] * 7,
                "record_time": pd.date_range("2025-05-01T11:00:00Z", periods=7, freq="1min"),
            }
        )
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
        df = pd.DataFrame(
            {
                "clip_id": ["a", "a"],
                "continuity_id": ["a:continuity-0001"] * 2,
                "record_time": pd.date_range("2025-05-01T11:00:00Z", periods=2, freq="1min"),
            }
        )
        probabilities = np.array([[0.95, 0.03, 0.02], [0.34, 0.33, 0.33]])
        result = apply_stable_state_policy(df, probabilities)
        assert result["traffic_state"].tolist() == [0, 0]
        assert bool(result["decision_abstained"].iloc[-1]) is True

    def test_state_memory_resets_for_each_clip(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["a", "a", "b"],
                "continuity_id": ["a:continuity-0001", "a:continuity-0001", "b:continuity-0001"],
                "record_time": pd.to_datetime(
                    ["2025-05-01T11:00:00Z", "2025-05-01T11:01:00Z", "2025-05-01T11:00:00Z"]
                ),
            }
        )
        probabilities = np.array(
            [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.02, 0.03, 0.95]]
        )
        result = apply_stable_state_policy(df, probabilities)
        assert result["traffic_state"].tolist() == [0, 0, 1]

    def test_state_memory_resets_after_a_long_gap(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["a"] * 4,
                "continuity_id": ["a:continuity-0001"] * 4,
                "record_time": pd.to_datetime(
                    [
                        "2025-05-01T11:00:00Z",
                        "2025-05-01T11:01:00Z",
                        "2025-05-01T11:02:00Z",
                        "2025-05-01T11:05:00Z",
                    ]
                ),
            }
        )
        probabilities = np.array(
            [[0.02, 0.03, 0.95], [0.02, 0.03, 0.95], [0.02, 0.03, 0.95], [0.95, 0.03, 0.02]]
        )

        result = apply_stable_state_policy(df, probabilities)

        assert result["traffic_state"].tolist() == [1, 1, 2, 0]
        assert result["continuity_id"].nunique() == 2


def test_progressive_batch_parity_detects_divergence() -> None:
    batch = classify_telemetry_dataframe(
        _feature_rows(),
        _DummyModel(np.array([0.9, 0.08, 0.02])),
        _DummyScaler(len(FEATURE_COLS)),
        model_revision="a" * 64,
    )
    progressive = batch[
        ["clip_id", "continuity_id", "record_time", "traffic_state", "state_label", "confidence"]
    ].copy()
    progressive["incident_candidate"] = batch["accident_rule_triggered"].astype(bool)

    assert_progressive_batch_parity(progressive, batch)
    progressive.loc[progressive.index[-1], "traffic_state"] = 1
    with pytest.raises(RuntimeError, match="diverged"):
        assert_progressive_batch_parity(progressive, batch)
