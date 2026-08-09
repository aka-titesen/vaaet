"""Explicit, composable training inputs for PostgreSQL, backups, and CSV bundles."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import pandas as pd

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.data.database import (
    DatabaseSettings,
    get_pg_restore_version,
    inspect_backup_catalog,
    load_human_ground_truth,
    load_telemetry,
    parse_sql_dump_tables,
    restore_backup_to_sql,
)
from vaaet.settings import FEATURE_COLS

DATASET_PACKAGE_CONTRACT = "vaaet-training-dataset-v1"
PACKAGE_FILES = {
    "raw": "raw-telemetry.csv",
    "features": "telemetry-features.csv",
    "predictions": "traffic-predictions.csv",
    "validations": "human-validations.csv",
}
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


TrainingSource = PostgresSource | PostgresBackupSource | RawCsvSource | DatasetPackageSource


@dataclass(frozen=True)
class TrainingIngestionPlan:
    raw_sources: tuple[TrainingSource, ...] = ()
    feedback_sources: tuple[TrainingSource, ...] = ()
    feedback_policy: FeedbackPolicy = FeedbackPolicy.VALIDATED_ONLY

    def __post_init__(self) -> None:
        if self.feedback_policy is not FeedbackPolicy.VALIDATED_ONLY:
            raise ValueError("Only human-validated feedback can be used for supervised training.")
        if not self.raw_sources and not self.feedback_sources:
            raise ValueError("At least one explicit training source is required.")


@dataclass(frozen=True)
class TrainingDataset:
    raw: pd.DataFrame
    validated_feedback: pd.DataFrame
    confirmed_incidents: pd.DataFrame
    provenance: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe path in dataset package: {member.filename}")
    archive.extractall(destination)


def create_dataset_package(
    output_path: str | Path,
    *,
    raw: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    validations: pd.DataFrame | None = None,
    provenance: dict[str, object] | None = None,
) -> Path:
    """Create a checksum-protected portable dataset package."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = {
        "raw": raw,
        "features": features,
        "predictions": predictions,
        "validations": validations,
    }
    if not any(frame is not None and not frame.empty for frame in frames.values()):
        raise ValueError("A dataset package must contain at least one non-empty table.")
    with tempfile.TemporaryDirectory(prefix="vaaet-dataset-") as temp_dir:
        root = Path(temp_dir)
        files: dict[str, dict[str, object]] = {}
        for key, frame in frames.items():
            if frame is None or frame.empty:
                continue
            filename = PACKAGE_FILES[key]
            path = root / filename
            frame.to_csv(path, index=False)
            files[key] = {
                "filename": filename,
                "rows": int(len(frame)),
                "sha256": _sha256(path),
                "columns": list(frame.columns),
            }
            if "record_time" in frame:
                timestamps = pd.to_datetime(frame["record_time"], utc=True, errors="raise")
                if timestamps.isna().any():
                    raise ValueError(f"Dataset component {key} contains missing record_time values.")
                files[key]["record_time_min"] = timestamps.min().isoformat()
                files[key]["record_time_max"] = timestamps.max().isoformat()
        manifest = {
            "contract_version": DATASET_PACKAGE_CONTRACT,
            "timezone": "UTC",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "files": files,
            "provenance": provenance or {},
        }
        (root / "dataset-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(root / "dataset-manifest.json", "dataset-manifest.json")
            for metadata in files.values():
                filename = str(metadata["filename"])
                archive.write(root / filename, filename)
    return output


def load_dataset_package(path: str | Path) -> dict[str, pd.DataFrame]:
    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"Dataset package not found: {package}")
    with tempfile.TemporaryDirectory(prefix="vaaet-dataset-read-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, root)
        manifest_path = root / "dataset-manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Dataset package is missing dataset-manifest.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_version") != DATASET_PACKAGE_CONTRACT:
            raise ValueError("Unsupported dataset package contract version.")
        frames: dict[str, pd.DataFrame] = {}
        for key, metadata in manifest.get("files", {}).items():
            if key not in PACKAGE_FILES or not isinstance(metadata, dict):
                raise ValueError(f"Unknown dataset package component: {key}")
            filename = metadata.get("filename")
            if filename != PACKAGE_FILES[key]:
                raise ValueError(f"Unexpected filename for package component {key}.")
            file_path = root / str(filename)
            if not file_path.is_file() or _sha256(file_path) != metadata.get("sha256"):
                raise ValueError(f"Checksum mismatch for dataset component {key}.")
            frame = pd.read_csv(file_path)
            if len(frame) != int(metadata.get("rows", -1)):
                raise ValueError(f"Row count mismatch for dataset component {key}.")
            if list(frame.columns) != metadata.get("columns"):
                raise ValueError(f"Column contract mismatch for dataset component {key}.")
            frames[key] = frame
        return frames


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
        frame = pd.read_csv(source.path)
    elif isinstance(source, DatasetPackageSource):
        frame = load_dataset_package(source.path).get("raw", pd.DataFrame())
    elif isinstance(source, PostgresBackupSource):
        frame = _frames_from_backup(source, components={"raw"}).get("raw", pd.DataFrame())
    else:
        raise TypeError(f"Unsupported raw source: {type(source)!r}")
    if frame.empty:
        details = frame.attrs.get("vaaet_provenance", {})
        archive_table = details.get("archive_table")
        suffix = f" ({archive_table})" if archive_table else ""
        raise ValueError(
            f"Explicit raw source {type(source).__name__}{suffix} contains zero telemetry rows."
        )
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
    raise TypeError(f"Unsupported feedback source: {type(source)!r}")


def _deduplicate_raw(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    required = RAW_REQUIRED_COLUMNS
    if missing := required - set(combined.columns):
        raise ValueError(f"Raw sources are missing fields: {sorted(missing)}")
    combined["record_time"] = pd.to_datetime(combined["record_time"], utc=True)
    comparison = [column for column in combined.columns if column not in {"id", "pipeline_run_id"}]
    for _, group in combined.groupby(["clip_id", "record_time"], dropna=False):
        if len(group[comparison].drop_duplicates()) > 1:
            raise ValueError(
                f"Conflicting raw records for clip={group.iloc[0]['clip_id']} "
                f"time={group.iloc[0]['record_time']}"
            )
    return combined.drop_duplicates(["clip_id", "record_time"], keep="last").reset_index(drop=True)


def _deduplicate_feedback(frames: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if "is_human_validated" in combined and not combined["is_human_validated"].fillna(False).all():
        raise ValueError("Feedback sources contain unvalidated predictions.")
    if "feature_schema_version" in combined:
        versions = set(combined["feature_schema_version"].dropna().astype(str))
        if versions and versions != {FEATURE_SCHEMA_VERSION}:
            raise ValueError(f"Incompatible feature schema versions: {sorted(versions)}")
    combined["record_time"] = pd.to_datetime(combined["record_time"], utc=True)
    combined["traffic_state"] = pd.to_numeric(combined["traffic_state"], errors="raise").astype(int)
    for _, group in combined.groupby(["clip_id", "record_time"], dropna=False):
        if group["traffic_state"].nunique() > 1 or len(
            group[[*FEATURE_COLS, "traffic_state"]].drop_duplicates()
        ) > 1:
            raise ValueError(
                f"Conflicting human labels or features for clip={group.iloc[0]['clip_id']} "
                f"time={group.iloc[0]['record_time']}"
            )
    combined = combined.drop_duplicates(["clip_id", "record_time"], keep="last")
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
    combined["record_time"] = pd.to_datetime(combined["record_time"], utc=True)
    return (
        combined.sort_values("is_human_validated")
        .drop_duplicates(["clip_id", "record_time"], keep="last")
        .sort_values(["clip_id", "record_time"])
        .reset_index(drop=True)
    )


def load_training_inputs(plan: TrainingIngestionPlan) -> TrainingDataset:
    raw_frames: list[pd.DataFrame] = []
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
    feedback, incidents = _deduplicate_feedback(feedback_frames)
    if raw.empty and feedback.empty and incidents.empty:
        raise ValueError("No usable raw telemetry or validated feedback was loaded.")
    return TrainingDataset(raw, feedback, incidents, pd.DataFrame(provenance))


__all__ = [
    "DATASET_PACKAGE_CONTRACT",
    "DatasetPackageSource",
    "FeedbackPolicy",
    "PostgresBackupSource",
    "PostgresSource",
    "RawCsvSource",
    "TrainingDataset",
    "TrainingIngestionPlan",
    "compose_supervised_dataset",
    "create_dataset_package",
    "load_dataset_package",
    "load_training_inputs",
]
