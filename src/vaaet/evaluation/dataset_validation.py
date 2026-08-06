"""Contractual dataset audit performed before any VAAET model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vaaet.data.datasets import TELEMETRY_QUALITY_COLUMNS, build_group_ids
from vaaet.settings import FEATURE_COLS, TELEMETRY_SCHEMA_VERSION

__all__ = ["DatasetAudit", "audit_training_dataset", "validate_training_partitions"]


@dataclass(frozen=True)
class DatasetAudit:
    report: dict[str, object]
    production_eligible: bool
    blockers: tuple[str, ...]


def audit_training_dataset(
    df: pd.DataFrame,
    *,
    require_production_eligible: bool = False,
) -> DatasetAudit:
    """Return provenance/quality evidence and stop on structural corruption."""
    required = {
        "clip_id",
        "record_time",
        "avg_speed",
        "total_vehicles",
        "count_car",
        "count_truck",
        "count_bus",
        "count_motorcycle",
        "count_bicycle",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset contract failed; missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset contract failed; no records were loaded.")

    timestamps = pd.to_datetime(df["record_time"], errors="raise")
    if df.duplicated(["clip_id", "record_time"]).any():
        raise ValueError("Dataset contract failed; duplicate clip/time records found.")
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
    speeds = pd.to_numeric(df["avg_speed"], errors="coerce")
    if speeds.isna().any() or speeds.lt(0).any() or speeds.gt(200).any():
        raise ValueError("Dataset contract failed; impossible avg_speed values found.")
    inconsistent_counts = int(counts.sum(axis=1).ne(total).sum())
    if inconsistent_counts:
        raise ValueError(
            f"Dataset contract failed; {inconsistent_counts} rows disagree with total_vehicles."
        )
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

    origins = df.get("data_origin", pd.Series("real", index=df.index)).fillna("unknown")
    real_mask = ~origins.eq("synthetic")
    real_frame = df.loc[real_mask]
    if real_frame.empty:
        raise ValueError("Dataset contract failed; no real telemetry is available.")
    modern_present = [column for column in TELEMETRY_QUALITY_COLUMNS if column in df]
    coverage = {
        column: round(float(real_frame[column].notna().mean()), 4) if column in df else 0.0
        for column in TELEMETRY_QUALITY_COLUMNS
    }
    schema = df.get("telemetry_schema_version", pd.Series(pd.NA, index=df.index))
    v2_coverage = float(schema.loc[real_mask].eq(TELEMETRY_SCHEMA_VERSION).mean())
    groups = build_group_ids(df)
    gaps = timestamps.groupby(groups).diff().dt.total_seconds().div(60)
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

    blockers: list[str] = []
    if v2_coverage < 0.95:
        blockers.append(
            f"telemetry v2 coverage is {v2_coverage:.1%}; at least 95% is required"
        )
    quality_fields = (
        "speed_measurement_quality",
        "optical_flow_tracking_ratio",
        "near_zero_motion_count",
        "stationary_confirmed_count",
    )
    incomplete = [field for field in quality_fields if coverage.get(field, 0.0) < 0.95]
    if incomplete:
        blockers.append(f"quality fields have insufficient coverage: {', '.join(incomplete)}")

    report: dict[str, object] = {
        "records": len(df),
        "clips": int(groups.nunique()),
        "time_start": timestamps.min().isoformat(),
        "time_end": timestamps.max().isoformat(),
        "records_by_origin": origins.value_counts(dropna=False).to_dict(),
        "records_by_schema": schema.fillna("traffic-telemetry-v1").value_counts().to_dict(),
        "modern_columns_present": modern_present,
        "modern_column_coverage": coverage,
        "telemetry_v2_coverage": round(v2_coverage, 4),
        "maximum_gap_minutes": None if gaps.dropna().empty else round(float(gaps.max()), 2),
        "median_frequency_minutes": None
        if gaps.dropna().empty
        else round(float(gaps.dropna().median()), 2),
        "numeric_summary": numeric_summary,
        "missing_feature_values": {
            column: int(df[column].isna().sum()) if column in df else len(df)
            for column in FEATURE_COLS
        },
        "production_blockers": blockers,
    }
    audit = DatasetAudit(report, not blockers, tuple(blockers))
    if require_production_eligible and blockers:
        raise ValueError("Dataset is not production-eligible: " + "; ".join(blockers))
    return audit


def validate_training_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Reject group leakage, synthetic evaluation, and unsupported labels."""
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
