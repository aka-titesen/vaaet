"""Tests for src/persistence.py — shared row preparation helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from vaaet.data.persistence import (
    _prepare_classification_row,
    _prepare_telemetry_row,
    persist_raw_telemetry,
)


def test_persist_raw_telemetry_is_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    telemetry = pd.DataFrame(
        [
            {
                "clip_id": "bridge_test",
                "record_time": pd.Timestamp("2025-05-01 08:01:00"),
                "avg_speed": 12.5,
                "count_car": 1,
                "count_truck": 0,
                "count_bus": 0,
                "count_motorcycle": 0,
                "count_bicycle": 0,
                "total_vehicles": 1,
            }
        ]
    )

    assert persist_raw_telemetry(telemetry, engine=engine) == 1
    assert persist_raw_telemetry(telemetry, engine=engine) == 0
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM traffic_data")).scalar_one() == 1


class TestPrepareTelemetryRow:
    def test_uses_id_as_source_record_id_when_needed(self) -> None:
        row = pd.Series(
            {
                "id": 7,
                "record_time": pd.Timestamp("2025-05-01 08:00:00"),
                "avg_speed": 12.5,
                "total_vehicles": 4,
            }
        )
        payload = _prepare_telemetry_row(row)
        assert payload["source_record_id"] == 7
        assert payload["speed_measurement_quality"] is None

    def test_handles_rows_without_source_record_id(self) -> None:
        row = pd.Series(
            {
                "record_time": pd.Timestamp("2025-05-01 08:00:00"),
                "avg_speed": 12.5,
                "total_vehicles": 4,
            }
        )
        payload = _prepare_telemetry_row(row)
        assert payload["source_record_id"] is None


class TestPrepareClassificationRow:
    def test_preserves_accident_gate_metadata(self) -> None:
        row = pd.Series(
            {
                "traffic_state": 3,
                "state_label": "Accident",
                "confidence": 0.91,
                "model_traffic_state": 2,
                "model_state_label": "Congested",
                "model_confidence": 0.64,
                "probability_margin": 0.31,
                "decision_abstained": False,
                "measurement_reliable": True,
                "accident_rule_triggered": True,
                "accident_gate_applied": True,
                "accident_evidence_score": 0.88,
                "is_human_validated": True,
                "human_override_state": 3,
            }
        )
        with pytest.raises(ValueError, match="automatic Accident"):
            _prepare_classification_row(row, telemetry_id=10, model_version="mlp-v2.0")

    def test_accepts_human_confirmed_accident_without_automatic_gate(self) -> None:
        row = pd.Series(
            {
                "traffic_state": 3,
                "state_label": "Accident",
                "confidence": 1.0,
                "model_traffic_state": 2,
                "accident_rule_triggered": True,
                "accident_gate_applied": False,
                "is_human_validated": True,
                "human_override_state": 3,
            }
        )
        payload = _prepare_classification_row(row, telemetry_id=10, model_version="mlp-v2.0")
        assert payload["traffic_state"] == 3
        assert payload["human_override_state"] == 3
        assert payload["accident_gate_applied"] is False
