"""Shared PostgreSQL persistence helpers for academic notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from vaaet.data.database import get_engine
from vaaet.logging import get_logger
from vaaet.settings import MODEL_VERSION, STATE_LABELS

logger = get_logger(__name__)

__all__ = [
    "PersistResult",
    "ensure_persistence_tables",
    "ensure_raw_telemetry_table",
    "persist_classified_telemetry",
    "persist_raw_telemetry",
]


DDL_TRAFFIC_DATA = """
CREATE TABLE IF NOT EXISTS traffic_data (
    id SERIAL PRIMARY KEY,
    clip_id TEXT NOT NULL,
    record_time TIMESTAMP NOT NULL,
    avg_speed NUMERIC(5,2) NOT NULL,
    count_car INTEGER NOT NULL,
    count_truck INTEGER NOT NULL,
    count_bus INTEGER NOT NULL,
    count_motorcycle INTEGER NOT NULL,
    count_bicycle INTEGER NOT NULL,
    total_vehicles INTEGER NOT NULL,
    near_zero_motion_count INTEGER,
    stationary_confirmed_count INTEGER,
    rejected_speed_count INTEGER,
    recovered_track_count INTEGER,
    speed_sample_count INTEGER,
    speed_measurement_quality NUMERIC(8,4),
    optical_flow_tracking_ratio NUMERIC(8,4),
    telemetry_schema_version TEXT,
    UNIQUE (clip_id, record_time)
);
"""

TRAFFIC_DATA_V2_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS near_zero_motion_count INTEGER;",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS stationary_confirmed_count INTEGER;",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS rejected_speed_count INTEGER;",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS recovered_track_count INTEGER;",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS speed_sample_count INTEGER;",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS speed_measurement_quality NUMERIC(8,4);",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS optical_flow_tracking_ratio NUMERIC(8,4);",
    "ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS telemetry_schema_version TEXT;",
)

INSERT_RAW_TELEMETRY_SQL = """
INSERT INTO traffic_data (
    clip_id, record_time, avg_speed, count_car, count_truck, count_bus,
    count_motorcycle, count_bicycle, total_vehicles,
    near_zero_motion_count, stationary_confirmed_count, rejected_speed_count,
    recovered_track_count, speed_sample_count, speed_measurement_quality,
    optical_flow_tracking_ratio, telemetry_schema_version
) VALUES (
    :clip_id, :record_time, :avg_speed, :count_car, :count_truck, :count_bus,
    :count_motorcycle, :count_bicycle, :total_vehicles,
    :near_zero_motion_count, :stationary_confirmed_count, :rejected_speed_count,
    :recovered_track_count, :speed_sample_count, :speed_measurement_quality,
    :optical_flow_tracking_ratio, :telemetry_schema_version
)
ON CONFLICT (clip_id, record_time) DO NOTHING;
"""


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
    optical_flow_tracking_ratio NUMERIC(8,4),
    near_zero_motion_ratio NUMERIC(8,4),
    stationary_confirmed_ratio NUMERIC(8,4),
    near_zero_motion_count INTEGER,
    stationary_confirmed_count INTEGER,
    rejected_speed_count INTEGER,
    recovered_track_count INTEGER,
    speed_sample_count INTEGER,
    telemetry_schema_version TEXT,
    data_origin TEXT,
    synthetic_scenario TEXT
);
"""

TELEMETRY_RAW_V2_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE telemetry_raw ADD COLUMN IF NOT EXISTS optical_flow_tracking_ratio NUMERIC(8,4);",
    "ALTER TABLE telemetry_raw ADD COLUMN IF NOT EXISTS telemetry_schema_version TEXT;",
)

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
    probability_margin NUMERIC(8,4),
    decision_abstained BOOLEAN DEFAULT FALSE,
    measurement_reliable BOOLEAN,
    accident_rule_triggered BOOLEAN DEFAULT FALSE,
    accident_alert_started BOOLEAN DEFAULT FALSE,
    accident_gate_applied BOOLEAN DEFAULT FALSE,
    accident_evidence_score NUMERIC(8,4),
    is_human_validated BOOLEAN DEFAULT FALSE,
    human_override_state SMALLINT,
    validated_at TIMESTAMP,
    UNIQUE (telemetry_id, model_version)
);
"""

CLASSIFICATION_V2_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE traffic_classifications ADD COLUMN IF NOT EXISTS probability_margin NUMERIC(8,4);",
    "ALTER TABLE traffic_classifications ADD COLUMN IF NOT EXISTS decision_abstained BOOLEAN DEFAULT FALSE;",
    "ALTER TABLE traffic_classifications ADD COLUMN IF NOT EXISTS measurement_reliable BOOLEAN;",
    "ALTER TABLE traffic_classifications ADD COLUMN IF NOT EXISTS accident_alert_started BOOLEAN DEFAULT FALSE;",
)

UPSERT_TELEMETRY_SQL = """
INSERT INTO telemetry_raw (
    source_record_id, clip_id, record_time, avg_speed, total_vehicles,
    count_car, count_truck, count_bus, count_motorcycle, count_bicycle,
    heavy_vehicle_ratio, delta_speed, delta_count, transition_flag,
    speed_variance, cumulative_delta_speed, low_speed_persistence,
    speed_measurement_quality, optical_flow_tracking_ratio, near_zero_motion_ratio,
    stationary_confirmed_ratio, near_zero_motion_count,
    stationary_confirmed_count, rejected_speed_count, recovered_track_count,
    speed_sample_count, telemetry_schema_version, data_origin, synthetic_scenario
) VALUES (
    :source_record_id, :clip_id, :record_time, :avg_speed, :total_vehicles,
    :count_car, :count_truck, :count_bus, :count_motorcycle, :count_bicycle,
    :heavy_vehicle_ratio, :delta_speed, :delta_count, :transition_flag,
    :speed_variance, :cumulative_delta_speed, :low_speed_persistence,
    :speed_measurement_quality, :optical_flow_tracking_ratio, :near_zero_motion_ratio,
    :stationary_confirmed_ratio, :near_zero_motion_count,
    :stationary_confirmed_count, :rejected_speed_count, :recovered_track_count,
    :speed_sample_count, :telemetry_schema_version, :data_origin, :synthetic_scenario
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
    optical_flow_tracking_ratio = EXCLUDED.optical_flow_tracking_ratio,
    near_zero_motion_ratio = EXCLUDED.near_zero_motion_ratio,
    stationary_confirmed_ratio = EXCLUDED.stationary_confirmed_ratio,
    near_zero_motion_count = EXCLUDED.near_zero_motion_count,
    stationary_confirmed_count = EXCLUDED.stationary_confirmed_count,
    rejected_speed_count = EXCLUDED.rejected_speed_count,
    recovered_track_count = EXCLUDED.recovered_track_count,
    speed_sample_count = EXCLUDED.speed_sample_count,
    telemetry_schema_version = EXCLUDED.telemetry_schema_version,
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
    speed_measurement_quality, optical_flow_tracking_ratio, near_zero_motion_ratio,
    stationary_confirmed_ratio, near_zero_motion_count,
    stationary_confirmed_count, rejected_speed_count, recovered_track_count,
    speed_sample_count, telemetry_schema_version, data_origin, synthetic_scenario
) VALUES (
    :source_record_id, :clip_id, :record_time, :avg_speed, :total_vehicles,
    :count_car, :count_truck, :count_bus, :count_motorcycle, :count_bicycle,
    :heavy_vehicle_ratio, :delta_speed, :delta_count, :transition_flag,
    :speed_variance, :cumulative_delta_speed, :low_speed_persistence,
    :speed_measurement_quality, :optical_flow_tracking_ratio, :near_zero_motion_ratio,
    :stationary_confirmed_ratio, :near_zero_motion_count,
    :stationary_confirmed_count, :rejected_speed_count, :recovered_track_count,
    :speed_sample_count, :telemetry_schema_version, :data_origin, :synthetic_scenario
)
RETURNING id;
"""

UPSERT_CLASSIFICATION_SQL = """
INSERT INTO traffic_classifications (
    telemetry_id, traffic_state, state_label, confidence, model_version,
    model_traffic_state, model_state_label, model_confidence,
    probability_margin, decision_abstained, measurement_reliable,
    accident_rule_triggered, accident_alert_started, accident_gate_applied, accident_evidence_score,
    is_human_validated, human_override_state
) VALUES (
    :telemetry_id, :traffic_state, :state_label, :confidence, :model_version,
    :model_traffic_state, :model_state_label, :model_confidence,
    :probability_margin, :decision_abstained, :measurement_reliable,
    :accident_rule_triggered, :accident_alert_started, :accident_gate_applied, :accident_evidence_score,
    :is_human_validated, :human_override_state
)
ON CONFLICT (telemetry_id, model_version) DO UPDATE SET
    traffic_state = EXCLUDED.traffic_state,
    state_label = EXCLUDED.state_label,
    confidence = EXCLUDED.confidence,
    model_traffic_state = EXCLUDED.model_traffic_state,
    model_state_label = EXCLUDED.model_state_label,
    model_confidence = EXCLUDED.model_confidence,
    probability_margin = EXCLUDED.probability_margin,
    decision_abstained = EXCLUDED.decision_abstained,
    measurement_reliable = EXCLUDED.measurement_reliable,
    accident_rule_triggered = EXCLUDED.accident_rule_triggered,
    accident_alert_started = EXCLUDED.accident_alert_started,
    accident_gate_applied = EXCLUDED.accident_gate_applied,
    accident_evidence_score = EXCLUDED.accident_evidence_score,
    is_human_validated = EXCLUDED.is_human_validated,
    human_override_state = EXCLUDED.human_override_state;
"""


@dataclass(frozen=True)
class PersistResult:
    telemetry_rows: int
    classification_rows: int


def _nullable_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _nullable_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _nullable_str(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def ensure_raw_telemetry_table(engine: Engine) -> None:
    """Create or migrate ``traffic_data`` to the nullable telemetry v2 schema."""
    with engine.begin() as conn:
        conn.execute(text(DDL_TRAFFIC_DATA))
        if engine.dialect.name == "postgresql":
            for statement in TRAFFIC_DATA_V2_MIGRATIONS:
                conn.execute(text(statement))


def persist_raw_telemetry(
    df: pd.DataFrame,
    *,
    config: dict[str, str] | None = None,
    engine: Engine | None = None,
) -> int:
    """Persist raw acquisition records idempotently into ``traffic_data``."""
    if df.empty:
        return 0

    required = {
        "clip_id",
        "record_time",
        "avg_speed",
        "count_car",
        "count_truck",
        "count_bus",
        "count_motorcycle",
        "count_bicycle",
        "total_vehicles",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw telemetry is missing required columns: {sorted(missing)}")

    owns_engine = engine is None
    active_engine = engine or get_engine(config)
    ensure_raw_telemetry_table(active_engine)
    inserted = 0
    try:
        with active_engine.begin() as conn:
            for row in df.itertuples(index=False):
                record_time = row.record_time
                if isinstance(record_time, pd.Timestamp):
                    record_time = record_time.to_pydatetime()
                payload = {
                    "clip_id": str(row.clip_id),
                    "record_time": record_time,
                    "avg_speed": float(row.avg_speed),
                    "count_car": int(row.count_car),
                    "count_truck": int(row.count_truck),
                    "count_bus": int(row.count_bus),
                    "count_motorcycle": int(row.count_motorcycle),
                    "count_bicycle": int(row.count_bicycle),
                    "total_vehicles": int(row.total_vehicles),
                    "near_zero_motion_count": _nullable_int(
                        getattr(row, "near_zero_motion_count", None)
                    ),
                    "stationary_confirmed_count": _nullable_int(
                        getattr(row, "stationary_confirmed_count", None)
                    ),
                    "rejected_speed_count": _nullable_int(
                        getattr(row, "rejected_speed_count", None)
                    ),
                    "recovered_track_count": _nullable_int(
                        getattr(row, "recovered_track_count", None)
                    ),
                    "speed_sample_count": _nullable_int(
                        getattr(row, "speed_sample_count", None)
                    ),
                    "speed_measurement_quality": _nullable_float(
                        getattr(row, "speed_measurement_quality", None)
                    ),
                    "optical_flow_tracking_ratio": _nullable_float(
                        getattr(row, "optical_flow_tracking_ratio", None)
                    ),
                    "telemetry_schema_version": _nullable_str(
                        getattr(row, "telemetry_schema_version", None)
                    ),
                }
                result = conn.execute(text(INSERT_RAW_TELEMETRY_SQL), payload)
                inserted += max(result.rowcount, 0)
    finally:
        if owns_engine:
            active_engine.dispose()
    return inserted


def ensure_persistence_tables(engine: Engine) -> None:
    """Create the academic persistence tables if they do not exist."""
    with engine.begin() as conn:
        conn.execute(text(DDL_TELEMETRY_RAW))
        conn.execute(text(DDL_TRAFFIC_CLASSIFICATIONS))
        if engine.dialect.name == "postgresql":
            for statement in TELEMETRY_RAW_V2_MIGRATIONS:
                conn.execute(text(statement))
            for statement in CLASSIFICATION_V2_MIGRATIONS:
                conn.execute(text(statement))


def _prepare_telemetry_row(row: pd.Series) -> dict[str, Any]:
    source_record_id = row.get("source_record_id", row.get("id"))
    if pd.isna(source_record_id):
        source_record_id = None

    return {
        "source_record_id": int(source_record_id) if source_record_id is not None else None,
        "clip_id": row.get("clip_id"),
        "record_time": row.get("record_time"),
        "avg_speed": float(row.get("avg_speed", 0.0))
        if pd.notna(row.get("avg_speed", 0.0))
        else None,
        "total_vehicles": int(row.get("total_vehicles", 0))
        if pd.notna(row.get("total_vehicles", 0))
        else None,
        "count_car": int(row.get("count_car", 0)) if pd.notna(row.get("count_car", 0)) else None,
        "count_truck": int(row.get("count_truck", 0))
        if pd.notna(row.get("count_truck", 0))
        else None,
        "count_bus": int(row.get("count_bus", 0)) if pd.notna(row.get("count_bus", 0)) else None,
        "count_motorcycle": int(row.get("count_motorcycle", 0))
        if pd.notna(row.get("count_motorcycle", 0))
        else None,
        "count_bicycle": int(row.get("count_bicycle", 0))
        if pd.notna(row.get("count_bicycle", 0))
        else None,
        "heavy_vehicle_ratio": float(row.get("heavy_vehicle_ratio", 0.0))
        if pd.notna(row.get("heavy_vehicle_ratio", 0.0))
        else None,
        "delta_speed": float(row.get("delta_speed", 0.0))
        if pd.notna(row.get("delta_speed", 0.0))
        else None,
        "delta_count": int(row.get("delta_count", 0))
        if pd.notna(row.get("delta_count", 0))
        else None,
        "transition_flag": int(row.get("transition_flag", 0))
        if pd.notna(row.get("transition_flag", 0))
        else 0,
        "speed_variance": float(row.get("speed_variance", 0.0))
        if pd.notna(row.get("speed_variance", 0.0))
        else None,
        "cumulative_delta_speed": float(row.get("cumulative_delta_speed", 0.0))
        if pd.notna(row.get("cumulative_delta_speed", 0.0))
        else None,
        "low_speed_persistence": float(row.get("low_speed_persistence", 0.0))
        if pd.notna(row.get("low_speed_persistence", 0.0))
        else None,
        "speed_measurement_quality": float(row.get("speed_measurement_quality"))
        if pd.notna(row.get("speed_measurement_quality"))
        else None,
        "optical_flow_tracking_ratio": float(row.get("optical_flow_tracking_ratio"))
        if pd.notna(row.get("optical_flow_tracking_ratio"))
        else None,
        "near_zero_motion_ratio": float(row.get("near_zero_motion_ratio"))
        if pd.notna(row.get("near_zero_motion_ratio"))
        else None,
        "stationary_confirmed_ratio": float(row.get("stationary_confirmed_ratio"))
        if pd.notna(row.get("stationary_confirmed_ratio"))
        else None,
        "near_zero_motion_count": int(row.get("near_zero_motion_count", 0))
        if pd.notna(row.get("near_zero_motion_count", 0))
        else 0,
        "stationary_confirmed_count": int(row.get("stationary_confirmed_count", 0))
        if pd.notna(row.get("stationary_confirmed_count", 0))
        else 0,
        "rejected_speed_count": int(row.get("rejected_speed_count", 0))
        if pd.notna(row.get("rejected_speed_count", 0))
        else 0,
        "recovered_track_count": int(row.get("recovered_track_count", 0))
        if pd.notna(row.get("recovered_track_count", 0))
        else 0,
        "speed_sample_count": int(row.get("speed_sample_count", 0))
        if pd.notna(row.get("speed_sample_count", 0))
        else 0,
        "telemetry_schema_version": _nullable_str(row.get("telemetry_schema_version")),
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
    is_human_validated = bool(row.get("is_human_validated", False))
    human_override = row.get("human_override_state")
    if traffic_state == 3 and not (
        is_human_validated and pd.notna(human_override) and int(human_override) == 3
    ):
        raise ValueError("Accident persistence requires a validated human override.")
    if bool(row.get("accident_gate_applied", False)):
        raise ValueError("Bundle v2 forbids an automatic Accident gate override.")
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
        "probability_margin": _nullable_float(row.get("probability_margin")),
        "decision_abstained": bool(row.get("decision_abstained", False)),
        "measurement_reliable": (
            bool(row.get("measurement_reliable"))
            if pd.notna(row.get("measurement_reliable"))
            else None
        ),
        "accident_rule_triggered": bool(row.get("accident_rule_triggered", False)),
        "accident_alert_started": bool(row.get("accident_alert_started", False)),
        "accident_gate_applied": bool(row.get("accident_gate_applied", False)),
        "accident_evidence_score": float(row.get("accident_evidence_score", 0.0)),
        "is_human_validated": is_human_validated,
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
