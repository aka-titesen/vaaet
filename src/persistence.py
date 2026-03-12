"""Shared PostgreSQL persistence helpers for academic notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import MODEL_VERSION, STATE_LABELS
from src.db import get_engine
from src.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "PersistResult",
    "ensure_persistence_tables",
    "persist_classified_telemetry",
]


DDL_TELEMETRY_RAW = """
CREATE TABLE IF NOT EXISTS telemetry_raw (
    id SERIAL PRIMARY KEY,
    source_record_id INTEGER UNIQUE,
    clip_id TEXT,
    record_time TIMESTAMP NOT NULL,
    avg_speed NUMERIC(8,2),
    total_vehicles INTEGER,
    count_car INTEGER,
    count_truck INTEGER,
    count_bus INTEGER,
    count_motorcycle INTEGER,
    count_bicycle INTEGER,
    heavy_vehicle_ratio NUMERIC(8,4),
    delta_speed NUMERIC(8,2),
    delta_count INTEGER,
    transition_flag SMALLINT DEFAULT 0,
    speed_variance NUMERIC(8,4),
    cumulative_delta_speed NUMERIC(8,2),
    low_speed_persistence NUMERIC(8,2),
    speed_measurement_quality NUMERIC(8,4),
    near_zero_motion_ratio NUMERIC(8,4),
    stationary_confirmed_ratio NUMERIC(8,4),
    near_zero_motion_count INTEGER,
    stationary_confirmed_count INTEGER,
    rejected_speed_count INTEGER,
    recovered_track_count INTEGER,
    speed_sample_count INTEGER,
    data_origin TEXT,
    synthetic_scenario TEXT
);
"""

DDL_TRAFFIC_CLASSIFICATIONS = """
CREATE TABLE IF NOT EXISTS traffic_classifications (
    id SERIAL PRIMARY KEY,
    telemetry_id INTEGER REFERENCES telemetry_raw(id),
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    traffic_state SMALLINT NOT NULL,
    state_label TEXT NOT NULL,
    confidence NUMERIC(8,4) NOT NULL,
    model_version TEXT NOT NULL,
    model_traffic_state SMALLINT,
    model_state_label TEXT,
    model_confidence NUMERIC(8,4),
    accident_rule_triggered BOOLEAN DEFAULT FALSE,
    accident_gate_applied BOOLEAN DEFAULT FALSE,
    accident_evidence_score NUMERIC(8,4),
    is_human_validated BOOLEAN DEFAULT FALSE,
    human_override_state SMALLINT,
    validated_at TIMESTAMP,
    UNIQUE (telemetry_id, model_version)
);
"""

UPSERT_TELEMETRY_SQL = """
INSERT INTO telemetry_raw (
    source_record_id, clip_id, record_time, avg_speed, total_vehicles,
    count_car, count_truck, count_bus, count_motorcycle, count_bicycle,
    heavy_vehicle_ratio, delta_speed, delta_count, transition_flag,
    speed_variance, cumulative_delta_speed, low_speed_persistence,
    speed_measurement_quality, near_zero_motion_ratio,
    stationary_confirmed_ratio, near_zero_motion_count,
    stationary_confirmed_count, rejected_speed_count, recovered_track_count,
    speed_sample_count, data_origin, synthetic_scenario
) VALUES (
    :source_record_id, :clip_id, :record_time, :avg_speed, :total_vehicles,
    :count_car, :count_truck, :count_bus, :count_motorcycle, :count_bicycle,
    :heavy_vehicle_ratio, :delta_speed, :delta_count, :transition_flag,
    :speed_variance, :cumulative_delta_speed, :low_speed_persistence,
    :speed_measurement_quality, :near_zero_motion_ratio,
    :stationary_confirmed_ratio, :near_zero_motion_count,
    :stationary_confirmed_count, :rejected_speed_count, :recovered_track_count,
    :speed_sample_count, :data_origin, :synthetic_scenario
)
ON CONFLICT (source_record_id) DO UPDATE SET
    clip_id = EXCLUDED.clip_id,
    record_time = EXCLUDED.record_time,
    avg_speed = EXCLUDED.avg_speed,
    total_vehicles = EXCLUDED.total_vehicles,
    count_car = EXCLUDED.count_car,
    count_truck = EXCLUDED.count_truck,
    count_bus = EXCLUDED.count_bus,
    count_motorcycle = EXCLUDED.count_motorcycle,
    count_bicycle = EXCLUDED.count_bicycle,
    heavy_vehicle_ratio = EXCLUDED.heavy_vehicle_ratio,
    delta_speed = EXCLUDED.delta_speed,
    delta_count = EXCLUDED.delta_count,
    transition_flag = EXCLUDED.transition_flag,
    speed_variance = EXCLUDED.speed_variance,
    cumulative_delta_speed = EXCLUDED.cumulative_delta_speed,
    low_speed_persistence = EXCLUDED.low_speed_persistence,
    speed_measurement_quality = EXCLUDED.speed_measurement_quality,
    near_zero_motion_ratio = EXCLUDED.near_zero_motion_ratio,
    stationary_confirmed_ratio = EXCLUDED.stationary_confirmed_ratio,
    near_zero_motion_count = EXCLUDED.near_zero_motion_count,
    stationary_confirmed_count = EXCLUDED.stationary_confirmed_count,
    rejected_speed_count = EXCLUDED.rejected_speed_count,
    recovered_track_count = EXCLUDED.recovered_track_count,
    speed_sample_count = EXCLUDED.speed_sample_count,
    data_origin = EXCLUDED.data_origin,
    synthetic_scenario = EXCLUDED.synthetic_scenario
RETURNING id;
"""

INSERT_TELEMETRY_SQL = """
INSERT INTO telemetry_raw (
    source_record_id, clip_id, record_time, avg_speed, total_vehicles,
    count_car, count_truck, count_bus, count_motorcycle, count_bicycle,
    heavy_vehicle_ratio, delta_speed, delta_count, transition_flag,
    speed_variance, cumulative_delta_speed, low_speed_persistence,
    speed_measurement_quality, near_zero_motion_ratio,
    stationary_confirmed_ratio, near_zero_motion_count,
    stationary_confirmed_count, rejected_speed_count, recovered_track_count,
    speed_sample_count, data_origin, synthetic_scenario
) VALUES (
    :source_record_id, :clip_id, :record_time, :avg_speed, :total_vehicles,
    :count_car, :count_truck, :count_bus, :count_motorcycle, :count_bicycle,
    :heavy_vehicle_ratio, :delta_speed, :delta_count, :transition_flag,
    :speed_variance, :cumulative_delta_speed, :low_speed_persistence,
    :speed_measurement_quality, :near_zero_motion_ratio,
    :stationary_confirmed_ratio, :near_zero_motion_count,
    :stationary_confirmed_count, :rejected_speed_count, :recovered_track_count,
    :speed_sample_count, :data_origin, :synthetic_scenario
)
RETURNING id;
"""

UPSERT_CLASSIFICATION_SQL = """
INSERT INTO traffic_classifications (
    telemetry_id, traffic_state, state_label, confidence, model_version,
    model_traffic_state, model_state_label, model_confidence,
    accident_rule_triggered, accident_gate_applied, accident_evidence_score,
    is_human_validated, human_override_state
) VALUES (
    :telemetry_id, :traffic_state, :state_label, :confidence, :model_version,
    :model_traffic_state, :model_state_label, :model_confidence,
    :accident_rule_triggered, :accident_gate_applied, :accident_evidence_score,
    :is_human_validated, :human_override_state
)
ON CONFLICT (telemetry_id, model_version) DO UPDATE SET
    traffic_state = EXCLUDED.traffic_state,
    state_label = EXCLUDED.state_label,
    confidence = EXCLUDED.confidence,
    model_traffic_state = EXCLUDED.model_traffic_state,
    model_state_label = EXCLUDED.model_state_label,
    model_confidence = EXCLUDED.model_confidence,
    accident_rule_triggered = EXCLUDED.accident_rule_triggered,
    accident_gate_applied = EXCLUDED.accident_gate_applied,
    accident_evidence_score = EXCLUDED.accident_evidence_score,
    is_human_validated = EXCLUDED.is_human_validated,
    human_override_state = EXCLUDED.human_override_state;
"""


@dataclass(frozen=True)
class PersistResult:
    telemetry_rows: int
    classification_rows: int


def ensure_persistence_tables(engine: Engine) -> None:
    """Create the academic persistence tables if they do not exist."""
    with engine.begin() as conn:
        conn.execute(text(DDL_TELEMETRY_RAW))
        conn.execute(text(DDL_TRAFFIC_CLASSIFICATIONS))


def _prepare_telemetry_row(row: pd.Series) -> dict[str, Any]:
    source_record_id = row.get("source_record_id", row.get("id"))
    if pd.isna(source_record_id):
        source_record_id = None

    return {
        "source_record_id": int(source_record_id) if source_record_id is not None else None,
        "clip_id": row.get("clip_id"),
        "record_time": row.get("record_time"),
        "avg_speed": float(row.get("avg_speed", 0.0)) if pd.notna(row.get("avg_speed", 0.0)) else None,
        "total_vehicles": int(row.get("total_vehicles", 0)) if pd.notna(row.get("total_vehicles", 0)) else None,
        "count_car": int(row.get("count_car", 0)) if pd.notna(row.get("count_car", 0)) else None,
        "count_truck": int(row.get("count_truck", 0)) if pd.notna(row.get("count_truck", 0)) else None,
        "count_bus": int(row.get("count_bus", 0)) if pd.notna(row.get("count_bus", 0)) else None,
        "count_motorcycle": int(row.get("count_motorcycle", 0)) if pd.notna(row.get("count_motorcycle", 0)) else None,
        "count_bicycle": int(row.get("count_bicycle", 0)) if pd.notna(row.get("count_bicycle", 0)) else None,
        "heavy_vehicle_ratio": float(row.get("heavy_vehicle_ratio", 0.0)) if pd.notna(row.get("heavy_vehicle_ratio", 0.0)) else None,
        "delta_speed": float(row.get("delta_speed", 0.0)) if pd.notna(row.get("delta_speed", 0.0)) else None,
        "delta_count": int(row.get("delta_count", 0)) if pd.notna(row.get("delta_count", 0)) else None,
        "transition_flag": int(row.get("transition_flag", 0)) if pd.notna(row.get("transition_flag", 0)) else 0,
        "speed_variance": float(row.get("speed_variance", 0.0)) if pd.notna(row.get("speed_variance", 0.0)) else None,
        "cumulative_delta_speed": float(row.get("cumulative_delta_speed", 0.0)) if pd.notna(row.get("cumulative_delta_speed", 0.0)) else None,
        "low_speed_persistence": float(row.get("low_speed_persistence", 0.0)) if pd.notna(row.get("low_speed_persistence", 0.0)) else None,
        "speed_measurement_quality": float(row.get("speed_measurement_quality", 1.0)) if pd.notna(row.get("speed_measurement_quality", 1.0)) else None,
        "near_zero_motion_ratio": float(row.get("near_zero_motion_ratio", 0.0)) if pd.notna(row.get("near_zero_motion_ratio", 0.0)) else None,
        "stationary_confirmed_ratio": float(row.get("stationary_confirmed_ratio", 0.0)) if pd.notna(row.get("stationary_confirmed_ratio", 0.0)) else None,
        "near_zero_motion_count": int(row.get("near_zero_motion_count", 0)) if pd.notna(row.get("near_zero_motion_count", 0)) else 0,
        "stationary_confirmed_count": int(row.get("stationary_confirmed_count", 0)) if pd.notna(row.get("stationary_confirmed_count", 0)) else 0,
        "rejected_speed_count": int(row.get("rejected_speed_count", 0)) if pd.notna(row.get("rejected_speed_count", 0)) else 0,
        "recovered_track_count": int(row.get("recovered_track_count", 0)) if pd.notna(row.get("recovered_track_count", 0)) else 0,
        "speed_sample_count": int(row.get("speed_sample_count", 0)) if pd.notna(row.get("speed_sample_count", 0)) else 0,
        "data_origin": row.get("data_origin", "real"),
        "synthetic_scenario": row.get("synthetic_scenario", "observed"),
    }


def _prepare_classification_row(
    row: pd.Series,
    *,
    telemetry_id: int,
    model_version: str,
) -> dict[str, Any]:
    traffic_state = int(row.get("traffic_state", 0))
    model_state = row.get("model_traffic_state", traffic_state)
    return {
        "telemetry_id": telemetry_id,
        "traffic_state": traffic_state,
        "state_label": row.get("state_label", STATE_LABELS[traffic_state]),
        "confidence": float(row.get("confidence", 0.0)),
        "model_version": row.get("model_version", model_version),
        "model_traffic_state": int(model_state) if pd.notna(model_state) else None,
        "model_state_label": row.get("model_state_label"),
        "model_confidence": float(row.get("model_confidence", row.get("confidence", 0.0))),
        "accident_rule_triggered": bool(row.get("accident_rule_triggered", False)),
        "accident_gate_applied": bool(row.get("accident_gate_applied", False)),
        "accident_evidence_score": float(row.get("accident_evidence_score", 0.0)),
        "is_human_validated": bool(row.get("is_human_validated", False)),
        "human_override_state": (
            int(row.get("human_override_state"))
            if pd.notna(row.get("human_override_state"))
            else None
        ),
    }


def persist_classified_telemetry(
    df: pd.DataFrame,
    *,
    config: dict[str, str] | None = None,
    engine: Engine | None = None,
    model_version: str = MODEL_VERSION,
) -> PersistResult:
    """Persist engineered telemetry and classifications with optional DB access."""
    if df.empty:
        logger.info("Skipping persistence because the dataframe is empty")
        return PersistResult(telemetry_rows=0, classification_rows=0)

    owns_engine = engine is None
    active_engine = engine or get_engine(config)
    ensure_persistence_tables(active_engine)

    telemetry_rows = 0
    classification_rows = 0

    try:
        with active_engine.begin() as conn:
            for row in df.itertuples(index=False):
                series = pd.Series(row._asdict())
                telemetry_payload = _prepare_telemetry_row(series)
                has_source_id = telemetry_payload["source_record_id"] is not None
                insert_sql = UPSERT_TELEMETRY_SQL if has_source_id else INSERT_TELEMETRY_SQL
                telemetry_id = conn.execute(text(insert_sql), telemetry_payload).scalar_one()
                telemetry_rows += 1

                classification_payload = _prepare_classification_row(
                    series,
                    telemetry_id=int(telemetry_id),
                    model_version=model_version,
                )
                conn.execute(text(UPSERT_CLASSIFICATION_SQL), classification_payload)
                classification_rows += 1
    finally:
        if owns_engine:
            active_engine.dispose()

    logger.info(
        "Persistence completed: telemetry_rows=%s classification_rows=%s model_version=%s",
        telemetry_rows,
        classification_rows,
        model_version,
    )
    return PersistResult(
        telemetry_rows=telemetry_rows,
        classification_rows=classification_rows,
    )
