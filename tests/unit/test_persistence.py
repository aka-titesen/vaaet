"""Tests for src/persistence.py — shared row preparation helpers."""

from __future__ import annotations

import pandas as pd

from src.persistence import _prepare_classification_row, _prepare_telemetry_row


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
        assert payload["speed_measurement_quality"] == 1.0

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
                "accident_rule_triggered": True,
                "accident_gate_applied": True,
                "accident_evidence_score": 0.88,
            }
        )
        payload = _prepare_classification_row(
            row,
            telemetry_id=10,
            model_version="mlp-v1.1",
        )
        assert payload["telemetry_id"] == 10
        assert payload["model_traffic_state"] == 2
        assert payload["accident_gate_applied"] is True
