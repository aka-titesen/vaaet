# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for schema-qualified PostgreSQL persistence payloads."""

from __future__ import annotations

import pandas as pd
import pytest

from vaaet_ml.data.persistence import (
    INSERT_RAW_SQL,
    UPSERT_FEATURE_SQL,
    UPSERT_PREDICTION_SQL,
    _feature_payload,
    _prediction_payload,
    _raw_payload,
)


def test_queries_use_versioned_schemas() -> None:
    assert "vaaet_raw.traffic_data" in INSERT_RAW_SQL
    assert "vaaet_ml.telemetry_features" in UPSERT_FEATURE_SQL
    assert "vaaet_ml.traffic_predictions" in UPSERT_PREDICTION_SQL
    assert "is_human_validated" not in UPSERT_PREDICTION_SQL
    assert "human_override_state" not in UPSERT_PREDICTION_SQL


def test_raw_payload_localizes_historical_buenos_aires_time() -> None:
    row = pd.Series(
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
    )
    payload = _raw_payload(row, "00000000-0000-0000-0000-000000000001")
    assert payload["record_time"].tzinfo is not None
    assert payload["record_time"].hour == 11


def test_feature_payload_uses_source_id_and_schema_version() -> None:
    row = pd.Series(
        {
            "id": 7,
            "clip_id": "clip",
            "record_time": pd.Timestamp("2025-05-01 08:00:00", tz="UTC"),
        }
    )
    payload = _feature_payload(row, "00000000-0000-0000-0000-000000000001")
    assert payload["source_record_id"] == 7
    assert payload["feature_schema_version"] == "traffic-features-v2"


def test_prediction_rejects_automatic_accident() -> None:
    row = pd.Series(
        {
            "traffic_state": 3,
            "model_traffic_state": 2,
            "state_label": "Accident",
            "confidence": 0.91,
        }
    )
    with pytest.raises(ValueError, match="exclusively"):
        _prediction_payload(
            row,
            feature_id=10,
            pipeline_run_id="00000000-0000-0000-0000-000000000001",
            model_version="mlp-v2.0",
        )


def test_prediction_preserves_incident_candidate_as_congested() -> None:
    row = pd.Series(
        {
            "traffic_state": 2,
            "state_label": "Congested",
            "confidence": 0.91,
            "model_traffic_state": 2,
            "accident_rule_triggered": True,
            "accident_alert_started": True,
            "accident_evidence_score": 0.88,
        }
    )
    payload = _prediction_payload(
        row,
        feature_id=10,
        pipeline_run_id="00000000-0000-0000-0000-000000000001",
        model_version="mlp-v2.0",
    )
    assert payload["traffic_state"] == 2
    assert payload["accident_rule_triggered"] is True
