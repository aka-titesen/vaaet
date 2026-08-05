"""Idempotent writes to the migrated VAAET PostgreSQL schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.data.database import DatabaseSettings, get_engine
from vaaet.logging import get_logger
from vaaet.settings import MODEL_VERSION, STATE_LABELS, TELEMETRY_SCHEMA_VERSION

logger = get_logger(__name__)

RAW_TABLE = "vaaet_raw.traffic_data"
FEATURE_TABLE = "vaaet_ml.telemetry_features"
PREDICTION_TABLE = "vaaet_ml.traffic_predictions"


@dataclass(frozen=True)
class PersistResult:
    telemetry_rows: int
    classification_rows: int
    pipeline_run_id: str


def _utc_timestamp(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/Argentina/Buenos_Aires")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _nullable_int(value: object) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _nullable_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _nullable_str(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _require_migrated_schema(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "The VAAET PostgreSQL schema is unavailable. Apply `alembic upgrade head` "
        "with the administrator profile before running a notebook."
    )


INSERT_RAW_SQL = f"""
INSERT INTO {RAW_TABLE} (
    pipeline_run_id, clip_id, record_time, avg_speed, count_car, count_truck,
    count_bus, count_motorcycle, count_bicycle, total_vehicles,
    near_zero_motion_count, stationary_confirmed_count, rejected_speed_count,
    recovered_track_count, speed_sample_count, speed_measurement_quality,
    optical_flow_tracking_ratio, telemetry_schema_version
) VALUES (
    :pipeline_run_id, :clip_id, :record_time, :avg_speed, :count_car, :count_truck,
    :count_bus, :count_motorcycle, :count_bicycle, :total_vehicles,
    :near_zero_motion_count, :stationary_confirmed_count, :rejected_speed_count,
    :recovered_track_count, :speed_sample_count, :speed_measurement_quality,
    :optical_flow_tracking_ratio, :telemetry_schema_version
)
ON CONFLICT (clip_id, record_time) DO NOTHING
"""


UPSERT_FEATURE_SQL = f"""
INSERT INTO {FEATURE_TABLE} (
    source_record_id, pipeline_run_id, clip_id, record_time, feature_schema_version,
    avg_speed, total_vehicles, count_car, count_truck, count_bus,
    count_motorcycle, count_bicycle, heavy_vehicle_ratio, delta_speed, delta_count,
    transition_flag, speed_variance, cumulative_delta_speed, low_speed_persistence,
    speed_measurement_quality, optical_flow_tracking_ratio, near_zero_motion_ratio,
    stationary_confirmed_ratio, near_zero_motion_count, stationary_confirmed_count,
    rejected_speed_count, recovered_track_count, speed_sample_count,
    telemetry_schema_version, data_origin, synthetic_scenario, hour_of_day,
    weather_condition
) VALUES (
    :source_record_id, :pipeline_run_id, :clip_id, :record_time, :feature_schema_version,
    :avg_speed, :total_vehicles, :count_car, :count_truck, :count_bus,
    :count_motorcycle, :count_bicycle, :heavy_vehicle_ratio, :delta_speed, :delta_count,
    :transition_flag, :speed_variance, :cumulative_delta_speed, :low_speed_persistence,
    :speed_measurement_quality, :optical_flow_tracking_ratio, :near_zero_motion_ratio,
    :stationary_confirmed_ratio, :near_zero_motion_count, :stationary_confirmed_count,
    :rejected_speed_count, :recovered_track_count, :speed_sample_count,
    :telemetry_schema_version, :data_origin, :synthetic_scenario, :hour_of_day,
    :weather_condition
)
ON CONFLICT (clip_id, record_time, feature_schema_version) DO UPDATE SET
    pipeline_run_id = EXCLUDED.pipeline_run_id,
    source_record_id = COALESCE({FEATURE_TABLE}.source_record_id, EXCLUDED.source_record_id),
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
    synthetic_scenario = EXCLUDED.synthetic_scenario,
    hour_of_day = EXCLUDED.hour_of_day,
    weather_condition = EXCLUDED.weather_condition
RETURNING id
"""


UPSERT_PREDICTION_SQL = f"""
INSERT INTO {PREDICTION_TABLE} (
    telemetry_feature_id, pipeline_run_id, classified_at, traffic_state, state_label,
    confidence, model_version, model_traffic_state, model_state_label,
    model_confidence, probability_margin, decision_abstained, measurement_reliable,
    accident_rule_triggered, accident_alert_started, accident_evidence_score
) VALUES (
    :telemetry_feature_id, :pipeline_run_id, CURRENT_TIMESTAMP, :traffic_state, :state_label,
    :confidence, :model_version, :model_traffic_state, :model_state_label,
    :model_confidence, :probability_margin, :decision_abstained, :measurement_reliable,
    :accident_rule_triggered, :accident_alert_started, :accident_evidence_score
)
ON CONFLICT (telemetry_feature_id, model_version) DO UPDATE SET
    pipeline_run_id = EXCLUDED.pipeline_run_id,
    classified_at = EXCLUDED.classified_at,
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
    accident_evidence_score = EXCLUDED.accident_evidence_score
"""


def _raw_payload(row: pd.Series, pipeline_run_id: str) -> dict[str, Any]:
    return {
        "pipeline_run_id": pipeline_run_id,
        "clip_id": str(row["clip_id"]),
        "record_time": _utc_timestamp(row["record_time"]),
        "avg_speed": float(row["avg_speed"]),
        "count_car": int(row["count_car"]),
        "count_truck": int(row["count_truck"]),
        "count_bus": int(row["count_bus"]),
        "count_motorcycle": int(row["count_motorcycle"]),
        "count_bicycle": int(row["count_bicycle"]),
        "total_vehicles": int(row["total_vehicles"]),
        "near_zero_motion_count": _nullable_int(row.get("near_zero_motion_count")),
        "stationary_confirmed_count": _nullable_int(row.get("stationary_confirmed_count")),
        "rejected_speed_count": _nullable_int(row.get("rejected_speed_count")),
        "recovered_track_count": _nullable_int(row.get("recovered_track_count")),
        "speed_sample_count": _nullable_int(row.get("speed_sample_count")),
        "speed_measurement_quality": _nullable_float(row.get("speed_measurement_quality")),
        "optical_flow_tracking_ratio": _nullable_float(row.get("optical_flow_tracking_ratio")),
        "telemetry_schema_version": _nullable_str(
            row.get("telemetry_schema_version", TELEMETRY_SCHEMA_VERSION)
        ),
    }


def _feature_payload(row: pd.Series, pipeline_run_id: str) -> dict[str, Any]:
    source_id = row.get("source_record_id", row.get("id"))
    return {
        "source_record_id": _nullable_int(source_id),
        "pipeline_run_id": pipeline_run_id,
        "clip_id": str(row["clip_id"]),
        "record_time": _utc_timestamp(row["record_time"]),
        "feature_schema_version": str(
            row.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        ),
        "avg_speed": _nullable_float(row.get("avg_speed")),
        "total_vehicles": _nullable_int(row.get("total_vehicles")),
        "count_car": _nullable_int(row.get("count_car")),
        "count_truck": _nullable_int(row.get("count_truck")),
        "count_bus": _nullable_int(row.get("count_bus")),
        "count_motorcycle": _nullable_int(row.get("count_motorcycle")),
        "count_bicycle": _nullable_int(row.get("count_bicycle")),
        "heavy_vehicle_ratio": _nullable_float(row.get("heavy_vehicle_ratio")),
        "delta_speed": _nullable_float(row.get("delta_speed")),
        "delta_count": _nullable_int(row.get("delta_count")),
        "transition_flag": _nullable_int(row.get("transition_flag")) or 0,
        "speed_variance": _nullable_float(row.get("speed_variance")),
        "cumulative_delta_speed": _nullable_float(row.get("cumulative_delta_speed")),
        "low_speed_persistence": _nullable_float(row.get("low_speed_persistence")),
        "speed_measurement_quality": _nullable_float(row.get("speed_measurement_quality")),
        "optical_flow_tracking_ratio": _nullable_float(row.get("optical_flow_tracking_ratio")),
        "near_zero_motion_ratio": _nullable_float(row.get("near_zero_motion_ratio")),
        "stationary_confirmed_ratio": _nullable_float(row.get("stationary_confirmed_ratio")),
        "near_zero_motion_count": _nullable_int(row.get("near_zero_motion_count")),
        "stationary_confirmed_count": _nullable_int(row.get("stationary_confirmed_count")),
        "rejected_speed_count": _nullable_int(row.get("rejected_speed_count")),
        "recovered_track_count": _nullable_int(row.get("recovered_track_count")),
        "speed_sample_count": _nullable_int(row.get("speed_sample_count")),
        "telemetry_schema_version": _nullable_str(row.get("telemetry_schema_version")),
        "data_origin": str(row.get("data_origin", "real")),
        "synthetic_scenario": str(row.get("synthetic_scenario", "observed")),
        "hour_of_day": _nullable_int(row.get("hour_of_day")),
        "weather_condition": _nullable_int(row.get("weather_condition")),
    }


def _prediction_payload(
    row: pd.Series,
    *,
    feature_id: int,
    pipeline_run_id: str,
    model_version: str,
) -> dict[str, Any]:
    state = int(row.get("traffic_state", 0))
    model_state = int(row.get("model_traffic_state", state))
    if state not in (0, 1, 2) or model_state not in (0, 1, 2):
        raise ValueError(
            "Automatic predictions may contain only Normal, Reduced, or Congested; "
            "Accident belongs exclusively to vaaet_feedback.human_validations."
        )
    if bool(row.get("accident_gate_applied", False)):
        raise ValueError("Bundle v2 forbids an automatic Accident gate override.")
    return {
        "telemetry_feature_id": feature_id,
        "pipeline_run_id": pipeline_run_id,
        "traffic_state": state,
        "state_label": str(row.get("state_label", STATE_LABELS[state])),
        "confidence": float(row.get("confidence", 0.0)),
        "model_version": str(row.get("model_version", model_version)),
        "model_traffic_state": model_state,
        "model_state_label": str(row.get("model_state_label", STATE_LABELS[model_state])),
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
        "accident_evidence_score": float(row.get("accident_evidence_score", 0.0)),
    }


def persist_raw_telemetry(
    df: pd.DataFrame,
    *,
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    config: Mapping[str, str] | None = None,
    engine: Engine | None = None,
    pipeline_run_id: UUID | str | None = None,
) -> int:
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
    run_id = str(pipeline_run_id or uuid4())
    owns_engine = engine is None
    active_engine = engine or get_engine(settings or config)
    inserted = 0
    try:
        with active_engine.begin() as connection:
            for _, row in df.iterrows():
                result = connection.execute(text(INSERT_RAW_SQL), _raw_payload(row, run_id))
                inserted += max(result.rowcount, 0)
    except ProgrammingError as exc:
        raise _require_migrated_schema(exc) from exc
    finally:
        if owns_engine:
            active_engine.dispose()
    return inserted


def persist_classified_telemetry(
    df: pd.DataFrame,
    *,
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    config: Mapping[str, str] | None = None,
    engine: Engine | None = None,
    model_version: str = MODEL_VERSION,
    pipeline_run_id: UUID | str | None = None,
) -> PersistResult:
    run_id = str(pipeline_run_id or uuid4())
    if df.empty:
        return PersistResult(0, 0, run_id)
    required = {"clip_id", "record_time", "traffic_state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Classified telemetry is missing required columns: {sorted(missing)}")
    owns_engine = engine is None
    active_engine = engine or get_engine(settings or config)
    telemetry_rows = 0
    prediction_rows = 0
    try:
        with active_engine.begin() as connection:
            for _, row in df.iterrows():
                feature_id = connection.execute(
                    text(UPSERT_FEATURE_SQL), _feature_payload(row, run_id)
                ).scalar_one()
                telemetry_rows += 1
                connection.execute(
                    text(UPSERT_PREDICTION_SQL),
                    _prediction_payload(
                        row,
                        feature_id=int(feature_id),
                        pipeline_run_id=run_id,
                        model_version=model_version,
                    ),
                )
                prediction_rows += 1
    except ProgrammingError as exc:
        raise _require_migrated_schema(exc) from exc
    finally:
        if owns_engine:
            active_engine.dispose()
    logger.info(
        "Persistence completed: telemetry_rows=%s prediction_rows=%s model_version=%s run=%s",
        telemetry_rows,
        prediction_rows,
        model_version,
        run_id,
    )
    return PersistResult(telemetry_rows, prediction_rows, run_id)


def ensure_raw_telemetry_table(engine: Engine) -> None:
    """Deprecated guard: schema changes are administrator-only in VAAET 4.1."""
    del engine
    raise RuntimeError("Apply `alembic upgrade head`; notebooks may not create database tables.")


def ensure_persistence_tables(engine: Engine) -> None:
    """Deprecated guard: schema changes are administrator-only in VAAET 4.1."""
    del engine
    raise RuntimeError("Apply `alembic upgrade head`; notebooks may not create database tables.")


__all__ = [
    "PersistResult",
    "ensure_persistence_tables",
    "ensure_raw_telemetry_table",
    "persist_classified_telemetry",
    "persist_raw_telemetry",
]
