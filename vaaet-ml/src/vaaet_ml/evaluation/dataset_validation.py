# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Auditoría contractual del dataset previa a todo entrenamiento VAAET."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.settings import (
    CANONICAL_TIMEZONE,
    FEATURE_COLS,
    TELEMETRY_SCHEMA_VERSION,
    TRAFFIC_LOCAL_TIMEZONE,
)
from vaaet.timestamps import normalize_timestamp_series

from vaaet_ml.data.datasets import TELEMETRY_QUALITY_COLUMNS, build_group_ids

__all__ = ["DatasetAudit", "audit_training_dataset", "validate_training_partitions"]


@dataclass(frozen=True)
class DatasetAudit:
    """Representa evidencia de calidad y bloqueos de elegibilidad del dataset."""

    report: dict[str, object]
    production_eligible: bool
    blockers: tuple[str, ...]


def audit_training_dataset(
    df: pd.DataFrame,
    *,
    require_production_eligible: bool = False,
) -> DatasetAudit:
    """Obtiene evidencia de procedencia y calidad; detiene datos corruptos."""
    _validate_dataset_source(df)
    df = df.copy()
    df["record_time"] = normalize_timestamp_series(df["record_time"])
    _validate_dataset_records(df)
    origins, real_mask = _resolve_data_origins(df)
    evidence = _build_audit_evidence(df, origins, real_mask)
    blockers = _production_blockers(evidence["coverage"], evidence["canonical_coverage"])
    report = _build_audit_report(df, origins, evidence, blockers)
    audit = DatasetAudit(report, not blockers, tuple(blockers))
    if require_production_eligible and blockers:
        raise ValueError("Dataset is not production-eligible: " + "; ".join(blockers))
    return audit


def _validate_dataset_source(df: pd.DataFrame) -> None:
    required = {
        "clip_id", "record_time", "avg_speed", "total_vehicles", "count_car", "count_truck",
        "count_bus", "count_motorcycle", "count_bicycle",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset contract failed; missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset contract failed; no records were loaded.")


def _validate_dataset_records(df: pd.DataFrame) -> None:
    if df.duplicated(["clip_id", "record_time"]).any():
        raise ValueError("Dataset contract failed; duplicate clip/time records found.")
    _validate_vehicle_counts(df)
    _validate_average_speed(df)
    _validate_quality_ranges(df)


def _validate_vehicle_counts(df: pd.DataFrame) -> None:
    counts = df[[
        "count_car",
        "count_truck",
        "count_bus",
        "count_motorcycle",
        "count_bicycle",
    ]].apply(pd.to_numeric, errors="coerce")
    total = pd.to_numeric(df["total_vehicles"], errors="coerce")
    if counts.isna().any().any() or total.isna().any():
        raise ValueError("Dataset contract failed; vehicle counts must be numeric.")
    if (counts < 0).any().any() or (total < 0).any():
        raise ValueError("Dataset contract failed; negative vehicle counts found.")
    inconsistent_counts = int(counts.sum(axis=1).ne(total).sum())
    if inconsistent_counts:
        raise ValueError(
            f"Dataset contract failed; {inconsistent_counts} rows disagree with total_vehicles."
        )


def _validate_average_speed(df: pd.DataFrame) -> None:
    speeds = pd.to_numeric(df["avg_speed"], errors="coerce")
    if speeds.isna().any() or speeds.lt(0).any() or speeds.gt(200).any():
        raise ValueError("Dataset contract failed; impossible avg_speed values found.")


def _validate_quality_ranges(df: pd.DataFrame) -> None:
    for ratio_column in ("speed_measurement_quality", "optical_flow_tracking_ratio"):
        if ratio_column in df:
            values = pd.to_numeric(df[ratio_column], errors="coerce").dropna()
            if not values.between(0.0, 1.0).all():
                raise ValueError(f"Dataset contract failed; {ratio_column} must be within [0,1].")
    for counter_column in (
        "near_zero_motion_count",
        "stationary_confirmed_count",
        "rejected_speed_count",
        "recovered_track_count",
        "speed_sample_count",
    ):
        if counter_column in df:
            values = pd.to_numeric(df[counter_column], errors="coerce").dropna()
            if values.lt(0).any():
                raise ValueError(f"Dataset contract failed; {counter_column} cannot be negative.")


def _resolve_data_origins(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    origins = df.get("data_origin", pd.Series("real", index=df.index)).fillna("unknown")
    real_mask = ~origins.eq("synthetic")
    if df.loc[real_mask].empty:
        raise ValueError("Dataset contract failed; no real telemetry is available.")
    return origins, real_mask


def _build_audit_evidence(
    df: pd.DataFrame, origins: pd.Series, real_mask: pd.Series
) -> dict[str, object]:
    """Calcula agregados trazables sin exponer filas ni alterar el dataset."""

    real_frame = df.loc[real_mask]
    processed = "feature_schema_version" in df
    quality_columns = (
        (
            "speed_measurement_quality",
            "near_zero_motion_ratio",
            "stationary_confirmed_ratio",
            "optical_flow_tracking_ratio",
        )
        if processed
        else TELEMETRY_QUALITY_COLUMNS
    )
    modern_present = [column for column in quality_columns if column in df]
    coverage = {
        column: round(float(real_frame[column].notna().mean()), 4) if column in df else 0.0
        for column in quality_columns
    }
    schema_column = "feature_schema_version" if processed else "telemetry_schema_version"
    expected_schema = FEATURE_SCHEMA_VERSION if processed else TELEMETRY_SCHEMA_VERSION
    schema = df.get(schema_column, pd.Series(pd.NA, index=df.index))
    canonical_coverage = float(schema.loc[real_mask].eq(expected_schema).mean())
    groups = build_group_ids(df)
    gaps = df["record_time"].groupby(groups).diff().dt.total_seconds().div(60)
    numeric_columns = [
        column
        for column in dict.fromkeys(["avg_speed", "total_vehicles", *FEATURE_COLS])
        if column in df and pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric_summary = {
        column: {
            "min": None if df[column].dropna().empty else float(df[column].min()),
            "p25": None if df[column].dropna().empty else float(df[column].quantile(0.25)),
            "median": None if df[column].dropna().empty else float(df[column].median()),
            "p75": None if df[column].dropna().empty else float(df[column].quantile(0.75)),
            "max": None if df[column].dropna().empty else float(df[column].max()),
        }
        for column in numeric_columns
    }
    return {
        "coverage": coverage,
        "gaps": gaps,
        "groups": groups,
        "modern_present": modern_present,
        "numeric_summary": numeric_summary,
        "schema": schema,
        "canonical_coverage": canonical_coverage,
    }


def _production_blockers(coverage: object, canonical_coverage: object) -> list[str]:
    if not isinstance(coverage, dict) or not isinstance(canonical_coverage, float):
        raise TypeError("Dataset audit evidence is malformed.")
    blockers: list[str] = []
    if canonical_coverage < 0.95:
        blockers.append(
            f"telemetry v3 coverage is {canonical_coverage:.1%}; at least 95% is required"
        )
    incomplete = [field for field, value in coverage.items() if value < 0.95]
    if incomplete:
        blockers.append(f"quality fields have insufficient coverage: {', '.join(incomplete)}")
    return blockers


def _build_audit_report(
    df: pd.DataFrame, origins: pd.Series, evidence: dict[str, object], blockers: list[str]
) -> dict[str, object]:
    groups = evidence["groups"]
    gaps = evidence["gaps"]
    schema = evidence["schema"]
    if not isinstance(groups, pd.Series) or not isinstance(gaps, pd.Series) or not isinstance(schema, pd.Series):
        raise TypeError("Dataset audit evidence is malformed.")

    return {
        "records": len(df),
        "clips": int(groups.nunique()),
        "time_start": df["record_time"].min().isoformat(),
        "time_end": df["record_time"].max().isoformat(),
        "timezone": CANONICAL_TIMEZONE,
        "traffic_local_timezone": TRAFFIC_LOCAL_TIMEZONE,
        "records_by_origin": origins.value_counts(dropna=False).to_dict(),
        "records_by_schema": schema.fillna("traffic-telemetry-v1").value_counts().to_dict(),
        "modern_columns_present": evidence["modern_present"],
        "modern_column_coverage": evidence["coverage"],
        "telemetry_v3_coverage": round(float(evidence["canonical_coverage"]), 4),
        "maximum_gap_minutes": None if gaps.dropna().empty else round(float(gaps.max()), 2),
        "median_frequency_minutes": None
        if gaps.dropna().empty
        else round(float(gaps.dropna().median()), 2),
        "numeric_summary": evidence["numeric_summary"],
        "missing_feature_values": {
            column: int(df[column].isna().sum()) if column in df else len(df)
            for column in FEATURE_COLS
        },
        "production_blockers": blockers,
    }


def validate_training_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Rechaza fuga entre grupos, evaluación sintética y etiquetas no admitidas."""
    partitions = {"train": train, "validation": validation, "test": test}
    groups = {name: set(build_group_ids(frame)) for name, frame in partitions.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"]:
        raise ValueError("A clip/group leaks from train into validation or test.")
    if groups["validation"] & groups["test"]:
        raise ValueError("A clip/group leaks between validation and test.")
    for name in ("validation", "test"):
        origins = partitions[name].get("data_origin", pd.Series("real", index=partitions[name].index))
        if origins.eq("synthetic").any():
            raise ValueError(f"Synthetic records are forbidden in {name}.")
    for name, frame in partitions.items():
        if "traffic_state" in frame and not set(frame["traffic_state"].dropna().astype(int)).issubset(
            {0, 1, 2}
        ):
            raise ValueError(f"{name} contains a label outside the three MLP output classes.")
