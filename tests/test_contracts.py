"""Tests for src/contracts.py — validated internal schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import (
    ClassificationRecord,
    EngineeredTelemetryRecord,
    TelemetryRecord,
    TrackSpeedState,
)


class TestTelemetryRecord:
    def test_from_mapping_coerces_timestamp_and_defaults(self) -> None:
        row = {
            "id": 1,
            "record_time": "2025-05-01 08:00:00",
            "avg_speed": 42.5,
            "count_car": 4,
            "count_truck": 1,
            "count_bus": 0,
            "count_motorcycle": 0,
            "count_bicycle": 0,
            "total_vehicles": 5,
            "near_zero_motion_count": 1,
            "stationary_confirmed_count": 0,
            "speed_measurement_quality": 0.8,
        }
        record = TelemetryRecord.from_mapping(row)
        assert isinstance(record.record_time, pd.Timestamp)
        assert record.data_origin == "real"
        assert record.synthetic_scenario == "observed"
        assert record.speed_measurement_quality == 0.8

    def test_invalid_origin_for_real_record_raises(self) -> None:
        with pytest.raises(ValueError):
            TelemetryRecord(
                id=1,
                record_time="2025-05-01 08:00:00",
                avg_speed=20.0,
                count_car=1,
                count_truck=0,
                count_bus=0,
                count_motorcycle=0,
                count_bicycle=0,
                total_vehicles=1,
                data_origin="real",
                synthetic_scenario="accident",
            )


class TestEngineeredTelemetryRecord:
    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            EngineeredTelemetryRecord(
                source_record_id=1,
                record_time="2025-05-01 08:00:00",
                avg_speed=20.0,
                total_vehicles=5,
                heavy_vehicle_ratio=1.5,
                delta_speed=-2.0,
                delta_count=1,
                transition_flag=0,
                speed_variance=1.0,
                cumulative_delta_speed=-3.0,
                low_speed_persistence=1.0,
                speed_measurement_quality=1.0,
                near_zero_motion_ratio=0.0,
                stationary_confirmed_ratio=0.0,
                hour_of_day=8,
                weather_condition=0,
            )


class TestClassificationRecord:
    def test_label_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ClassificationRecord(
                telemetry_id=1,
                traffic_state=3,
                state_label="Congested",
                confidence=0.8,
                model_version="mlp-v1.1",
                data_origin="synthetic",
                synthetic_scenario="accident",
            )

    def test_valid_synthetic_accident_record(self) -> None:
        record = ClassificationRecord(
            telemetry_id=1,
            traffic_state=3,
            state_label="Accident",
            confidence=0.8,
            model_version="mlp-v1.1",
            data_origin="synthetic",
            synthetic_scenario="accident",
            model_traffic_state=2,
            model_confidence=0.63,
            accident_rule_triggered=True,
            accident_gate_applied=True,
            accident_evidence_score=0.9,
        )
        assert record.traffic_state == 3
        assert record.accident_gate_applied is True


class TestTrackSpeedState:
    def test_invalid_vehicle_type_raises(self) -> None:
        with pytest.raises(ValueError):
            TrackSpeedState(
                track_id=1,
                vehicle_type="train",
                history_length=10,
                flow_tracking_ratio=1.0,
                recovered_after_gap=0,
                is_stationary=False,
            )

    def test_candidate_speed_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            TrackSpeedState(
                track_id=1,
                vehicle_type="car",
                history_length=10,
                flow_tracking_ratio=0.9,
                recovered_after_gap=0,
                is_stationary=False,
                candidate_speed=-1.0,
            )

    def test_measurement_quality_must_be_ratio(self) -> None:
        with pytest.raises(ValueError):
            TrackSpeedState(
                track_id=1,
                vehicle_type="car",
                history_length=10,
                flow_tracking_ratio=0.9,
                recovered_after_gap=0,
                is_stationary=False,
                measurement_quality=1.5,
            )
