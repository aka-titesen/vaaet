# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit, composable training inputs for PostgreSQL, backups, and CSV bundles."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.settings import FEATURE_COLS
from vaaet.timestamps import (
    count_naive_timestamps,
    normalize_timestamp_series,
)

from vaaet_ml.data.database import (
    DatabaseSettings,
    get_pg_restore_version,
    inspect_backup_catalog,
    load_human_ground_truth,
    load_telemetry,
    parse_sql_dump_tables,
    restore_backup_to_sql,
)
from vaaet_ml.data.dataset_artifacts import HitlCatalogSource, load_hitl_catalog_feedback
from vaaet_ml.data.package_codec import (
    DATASET_PACKAGE_CONTRACT,
    SEED_DATASET_PACKAGE_CONTRACT,
    create_dataset_package,
    load_dataset_package,
)
from vaaet_ml.training.lifecycle import TrainingMode

RAW_REQUIRED_COLUMNS = {
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


class FeedbackPolicy(str, Enum):
    VALIDATED_ONLY = "validated_only"


@dataclass(frozen=True)
class PostgresSource:
    settings: DatabaseSettings


@dataclass(frozen=True)
class PostgresBackupSource:
    path: Path
    pg_restore_path: Path | None = None


@dataclass(frozen=True)
class RawCsvSource:
    path: Path


@dataclass(frozen=True)
class DatasetPackageSource:
    path: Path


@dataclass(frozen=True)
class SeedDatasetPackageSource:
    """Processed weak-label seed package; never inferred from generic feature files."""

    path: Path


TrainingSource = (
    PostgresSource
    | PostgresBackupSource
    | RawCsvSource
    | DatasetPackageSource
    | SeedDatasetPackageSource
    | HitlCatalogSource
)


@dataclass(frozen=True)
class TrainingIngestionPlan:
    mode: TrainingMode
    raw_sources: tuple[TrainingSource, ...] = ()
    seed_sources: tuple[SeedDatasetPackageSource, ...] = ()
    feedback_sources: tuple[TrainingSource, ...] = ()
    feedback_policy: FeedbackPolicy = FeedbackPolicy.VALIDATED_ONLY

    def __post_init__(self) -> None:
        if self.feedback_policy is not FeedbackPolicy.VALIDATED_ONLY:
            raise ValueError("Only human-validated feedback can be used for supervised training.")
        if not self.raw_sources and not self.seed_sources and not self.feedback_sources:
            raise ValueError("At least one explicit training source is required.")
        if self.mode is TrainingMode.HITL_RETRAINING and not self.feedback_sources:
            raise ValueError("HITL retraining requires an explicit validated feedback source.")


@dataclass(frozen=True)
class TrainingDataset:
    raw: pd.DataFrame
    seed_features: pd.DataFrame
    validated_feedback: pd.DataFrame
    confirmed_incidents: pd.DataFrame
    provenance: pd.DataFrame


def _load_seed_features(source: SeedDatasetPackageSource) -> pd.DataFrame:
    frames = load_dataset_package(
        source.path,
        accepted_contracts=(SEED_DATASET_PACKAGE_CONTRACT, DATASET_PACKAGE_CONTRACT),
    )
    frame = frames.get("features", pd.DataFrame())
    if frame.empty:
        raise ValueError("Explicit seed package contains zero processed feature rows.")
    required = {"clip_id", "record_time", "traffic_state", *FEATURE_COLS}
    if missing := required - set(frame.columns):
        raise ValueError(f"Seed feature package is missing fields: {sorted(missing)}")
    feature_order = [column for column in frame.columns if column in FEATURE_COLS]
    if feature_order != list(FEATURE_COLS):
        raise ValueError("Seed package does not preserve the exact 19-feature order.")
    if not pd.to_numeric(frame["traffic_state"], errors="raise").isin((0, 1, 2)).all():
        raise ValueError("Seed package may contain only stable proxy labels 0, 1, and 2.")
    if "is_human_validated" in frame and frame["is_human_validated"].fillna(False).any():
        raise ValueError("Seed package cannot contain human-validated rows.")
    if "feature_schema_version" in frame:
        versions = set(frame["feature_schema_version"].dropna().astype(str))
        if versions and versions != {FEATURE_SCHEMA_VERSION}:
            raise ValueError(f"Incompatible seed feature schema versions: {sorted(versions)}")
    package_provenance = frame.attrs.get("vaaet_package_provenance", {})
    package_contract = frame.attrs.get("vaaet_package_contract")
    if package_provenance.get("training_mode") != TrainingMode.SEED_BOOTSTRAP.value or (
        package_provenance.get("supervision") != "weak-proxy"
    ):
        raise ValueError(
            "Seed package provenance must declare seed-bootstrap weak-proxy supervision."
        )
    if package_contract == DATASET_PACKAGE_CONTRACT:
        warnings.warn(
            "Legacy seed package uses vaaet-training-dataset-v1; register it in the "
            "versioned seed store to migrate to vaaet-seed-bootstrap-v1.",
            DeprecationWarning,
            stacklevel=2,
        )
    frame = frame.copy()
    frame["record_time"] = normalize_timestamp_series(frame["record_time"])
    frame["traffic_state"] = pd.to_numeric(frame["traffic_state"], errors="raise").astype(int)
    frame["is_human_validated"] = False
    frame.attrs["vaaet_provenance"] = {
        "package_kind": "processed-seed",
        **package_provenance,
    }
    return frame


def _latest_validated_feedback(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    features = frames.get("features", pd.DataFrame())
    predictions = frames.get("predictions", pd.DataFrame())
    validations = frames.get("validations", pd.DataFrame())
    if features.empty or predictions.empty or validations.empty:
        return pd.DataFrame()
    for name, frame, required in (
        ("features", features, {"id", "clip_id", "record_time", *FEATURE_COLS}),
        ("predictions", predictions, {"id", "telemetry_feature_id", "model_version"}),
        ("validations", validations, {"prediction_id", "validated_state", "reviewed_at"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} component is missing fields: {sorted(missing)}")
    validations = validations.copy()
    validations["reviewed_at"] = pd.to_datetime(validations["reviewed_at"], utc=True)
    ordering = ["reviewed_at", "id"] if "id" in validations else ["reviewed_at"]
    latest = validations.sort_values(ordering).drop_duplicates("prediction_id", keep="last")
    joined = features.merge(
        predictions[["id", "telemetry_feature_id", "model_version"]],
        left_on="id",
        right_on="telemetry_feature_id",
        suffixes=("", "_prediction"),
    ).merge(latest, left_on="id_prediction", right_on="prediction_id")
    joined["traffic_state"] = pd.to_numeric(joined["validated_state"], errors="raise").astype(int)
    joined["is_human_validated"] = True
    return joined


def _frames_from_backup(
    source: PostgresBackupSource, *, components: set[str]
) -> dict[str, pd.DataFrame]:
    catalog = inspect_backup_catalog(source.path, pg_restore_path=source.pg_restore_path)
    if not catalog:
        raise ValueError("PostgreSQL backup contains no recognized VAAET tables.")
    aliases = {
        "raw": ("vaaet_raw.traffic_data", "public.traffic_data"),
        "features": ("vaaet_ml.telemetry_features", "public.telemetry_raw"),
        "predictions": ("vaaet_ml.traffic_predictions", "public.traffic_classifications"),
        "validations": ("vaaet_feedback.human_validations",),
    }
    unknown = components - set(aliases)
    if unknown:
        raise ValueError(f"Unknown backup components requested: {sorted(unknown)}")
    selected_tables = tuple(
        table_name
        for component in components
        for table_name in aliases[component]
        if table_name in catalog
    )
    if not selected_tables:
        raise ValueError(
            f"PostgreSQL backup does not contain requested components: {sorted(components)}"
        )
    reader_version = get_pg_restore_version(source.pg_restore_path)
    sql_path = restore_backup_to_sql(
        source.path,
        pg_restore_path=source.pg_restore_path,
        tables=selected_tables,
    )
    try:
        tables = parse_sql_dump_tables(sql_path)
    finally:
        sql_path.unlink(missing_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for key, names in aliases.items():
        for name in names:
            if name in tables:
                frame = tables[name]
                frame.attrs["vaaet_provenance"] = {
                    "archive_table": name,
                    "backup_layout": "legacy" if name.startswith("public.") else "modern",
                    "reader_version": reader_version,
                }
                result[key] = frame
                break
    if "validations" not in result and "predictions" in result:
        legacy = result["predictions"]
        if "is_human_validated" in legacy:
            validated = legacy.loc[legacy["is_human_validated"].fillna(False).astype(bool)].copy()
            if not validated.empty:
                validated["prediction_id"] = validated["id"]
                validated["validated_state"] = validated["human_override_state"].fillna(
                    validated["traffic_state"]
                )
                validated["reviewed_at"] = validated.get("validated_at", validated["classified_at"])
                validated["reviewer_id"] = "legacy-import"
                validated.attrs["vaaet_provenance"] = legacy.attrs.get(
                    "vaaet_provenance", {}
                )
                result["validations"] = validated
    return result


def _load_raw(source: TrainingSource) -> pd.DataFrame:
    if isinstance(source, PostgresSource):
        frame = load_telemetry(settings=source.settings)
    elif isinstance(source, RawCsvSource):
        frame = pd.read_csv(source.path, float_precision="round_trip")
    elif isinstance(source, DatasetPackageSource):
        frame = load_dataset_package(source.path).get("raw", pd.DataFrame())
    elif isinstance(source, PostgresBackupSource):
        frame = _frames_from_backup(source, components={"raw"}).get("raw", pd.DataFrame())
    elif isinstance(source, SeedDatasetPackageSource):
        raise ValueError("SeedDatasetPackageSource must be declared in seed_sources.")
    elif isinstance(source, HitlCatalogSource):
        raise ValueError("HitlCatalogSource cannot be used as a raw source.")
    else:
        raise TypeError(f"Unsupported raw source: {type(source)!r}")
    if frame.empty:
        details = frame.attrs.get("vaaet_provenance", {})
        archive_table = details.get("archive_table")
        suffix = f" ({archive_table})" if archive_table else ""
        raise ValueError(
            f"Explicit raw source {type(source).__name__}{suffix} contains zero telemetry rows."
        )
    temporal_provenance = dict(frame.attrs.get("vaaet_provenance", {}))
    naive_count = count_naive_timestamps(frame["record_time"])
    frame = frame.copy()
    frame["record_time"] = normalize_timestamp_series(frame["record_time"])
    temporal_provenance.update(
        {
            "timestamp_timezone": "UTC",
            "naive_timezone_assumption": "America/Argentina/Buenos_Aires",
            "naive_timestamps_localized": naive_count,
        }
    )
    frame.attrs["vaaet_provenance"] = temporal_provenance
    return frame


def _load_feedback(source: TrainingSource) -> pd.DataFrame:
    if isinstance(source, PostgresSource):
        return load_human_ground_truth(settings=source.settings)
    if isinstance(source, DatasetPackageSource):
        return _latest_validated_feedback(load_dataset_package(source.path))
    if isinstance(source, PostgresBackupSource):
        frames = _frames_from_backup(
            source, components={"features", "predictions", "validations"}
        )
        frame = _latest_validated_feedback(frames)
        source_details = next(
            (
                item.attrs.get("vaaet_provenance", {})
                for item in frames.values()
                if item.attrs.get("vaaet_provenance")
            ),
            {},
        )
        frame.attrs["vaaet_provenance"] = source_details
        return frame
    if isinstance(source, RawCsvSource):
        raise ValueError("RawCsvSource cannot be used as a feedback source.")
    if isinstance(source, SeedDatasetPackageSource):
        raise ValueError("SeedDatasetPackageSource cannot be used as a feedback source.")
    if isinstance(source, HitlCatalogSource):
        frame, descriptor = load_hitl_catalog_feedback(source)
        frame.attrs["vaaet_provenance"] = descriptor
        return frame
    raise TypeError(f"Unsupported feedback source: {type(source)!r}")


def _deduplicate_raw(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    required = RAW_REQUIRED_COLUMNS
    if missing := required - set(combined.columns):
        raise ValueError(f"Raw sources are missing fields: {sorted(missing)}")
    combined["record_time"] = normalize_timestamp_series(combined["record_time"])
    comparison = [column for column in combined.columns if column not in {"id", "pipeline_run_id"}]
    for _, group in combined.groupby(["clip_id", "record_time"], dropna=False):
        if len(group[comparison].drop_duplicates()) > 1:
            raise ValueError(
                f"Conflicting raw records for clip={group.iloc[0]['clip_id']} "
                f"time={group.iloc[0]['record_time']}"
            )
    return combined.drop_duplicates(["clip_id", "record_time"], keep="last").reset_index(drop=True)


def _deduplicate_feedback(
    frames: Sequence[pd.DataFrame], *, require_human_validation: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_empty = [frame.copy() for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    required = {"clip_id", "record_time", "traffic_state", *FEATURE_COLS}
    if missing := required - set(combined.columns):
        raise ValueError(f"Validated feedback is missing fields: {sorted(missing)}")
    feature_order = [column for column in combined.columns if column in FEATURE_COLS]
    if feature_order != list(FEATURE_COLS):
        raise ValueError("Validated feedback does not preserve the exact 19-feature order.")
    if (
        require_human_validation
        and "is_human_validated" in combined
        and not combined["is_human_validated"].fillna(False).all()
    ):
        raise ValueError("Feedback sources contain unvalidated predictions.")
    if "feature_schema_version" in combined:
        versions = set(combined["feature_schema_version"].dropna().astype(str))
        if versions and versions != {FEATURE_SCHEMA_VERSION}:
            raise ValueError(f"Incompatible feature schema versions: {sorted(versions)}")
    combined["record_time"] = normalize_timestamp_series(combined["record_time"])
    combined["traffic_state"] = pd.to_numeric(combined["traffic_state"], errors="raise").astype(int)
    for _, group in combined.groupby(["clip_id", "record_time"], dropna=False):
        if group["traffic_state"].nunique() > 1 or len(
            group[[*FEATURE_COLS, "traffic_state"]].drop_duplicates()
        ) > 1:
            raise ValueError(
                f"Conflicting human labels or features for clip={group.iloc[0]['clip_id']} "
                f"time={group.iloc[0]['record_time']}"
            )
    combined = (
        combined.drop_duplicates(["clip_id", "record_time"], keep="last")
        .sort_values(["clip_id", "record_time"])
        .reset_index(drop=True)
    )
    incidents = combined.loc[combined["traffic_state"].eq(3)].reset_index(drop=True)
    stable = combined.loc[combined["traffic_state"].isin((0, 1, 2))].reset_index(drop=True)
    return stable, incidents


def compose_supervised_dataset(
    proxy_features: pd.DataFrame, validated_feedback: pd.DataFrame
) -> pd.DataFrame:
    """Combine stable proxy labels with human labels, giving humans precedence."""
    if proxy_features.empty and validated_feedback.empty:
        raise ValueError("No stable training records are available for composition.")
    frames: list[pd.DataFrame] = []
    if not proxy_features.empty:
        proxy = proxy_features.copy()
        proxy["is_human_validated"] = False
        frames.append(proxy)
    if not validated_feedback.empty:
        human = validated_feedback.copy()
        if not human["traffic_state"].isin((0, 1, 2)).all():
            raise ValueError("Accident must not enter the stable MLP supervised dataset.")
        human["data_origin"] = "real"
        human["synthetic_scenario"] = "observed"
        human["is_human_validated"] = True
        frames.append(human)
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    aligned = [frame.reindex(columns=columns) for frame in frames]
    combined = pd.concat(aligned, ignore_index=True)
    combined["record_time"] = normalize_timestamp_series(combined["record_time"])
    return (
        combined.sort_values("is_human_validated")
        .drop_duplicates(["clip_id", "record_time"], keep="last")
        .sort_values(["clip_id", "record_time"])
        .reset_index(drop=True)
    )


def load_training_inputs(plan: TrainingIngestionPlan) -> TrainingDataset:
    raw_frames: list[pd.DataFrame] = []
    seed_frames: list[pd.DataFrame] = []
    feedback_frames: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for index, source in enumerate(plan.raw_sources):
        frame = _load_raw(source)
        raw_frames.append(frame)
        provenance.append({
            "kind": "raw",
            "source_index": index,
            "source_type": type(source).__name__,
            "rows": len(frame),
            **frame.attrs.get("vaaet_provenance", {}),
        })
    for index, source in enumerate(plan.seed_sources):
        frame = _load_seed_features(source)
        seed_frames.append(frame)
        provenance.append({
            "kind": "processed_seed",
            "source_index": index,
            "source_type": type(source).__name__,
            "rows": len(frame),
            **frame.attrs.get("vaaet_provenance", {}),
        })
    for index, source in enumerate(plan.feedback_sources):
        frame = _load_feedback(source)
        feedback_frames.append(frame)
        provenance.append({
            "kind": "validated_feedback",
            "source_index": index,
            "source_type": type(source).__name__,
            "rows": len(frame),
            **frame.attrs.get("vaaet_provenance", {}),
        })
    raw = _deduplicate_raw(raw_frames)
    seed, seed_incidents = _deduplicate_feedback(
        seed_frames, require_human_validation=False
    )
    if not seed_incidents.empty:
        raise ValueError("Processed seed datasets cannot contain Accident targets.")
    feedback, incidents = _deduplicate_feedback(feedback_frames)
    if raw.empty and seed.empty and feedback.empty and incidents.empty:
        raise ValueError("No usable raw telemetry or validated feedback was loaded.")
    return TrainingDataset(raw, seed, feedback, incidents, pd.DataFrame(provenance))


__all__ = [
    "DATASET_PACKAGE_CONTRACT",
    "SEED_DATASET_PACKAGE_CONTRACT",
    "DatasetPackageSource",
    "FeedbackPolicy",
    "HitlCatalogSource",
    "PostgresBackupSource",
    "PostgresSource",
    "RawCsvSource",
    "SeedDatasetPackageSource",
    "TrainingDataset",
    "TrainingIngestionPlan",
    "compose_supervised_dataset",
    "create_dataset_package",
    "load_dataset_package",
    "load_training_inputs",
]
