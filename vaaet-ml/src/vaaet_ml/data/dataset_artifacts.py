"""Immutable dataset snapshots, HITL catalogs, and reproducible training input locks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.timestamps import normalize_timestamp_series

from vaaet_ml.settings import FEATURE_COLS, MODEL_STATE_LABELS

SEED_ARTIFACT_CONTRACT = "vaaet-seed-bootstrap-v1"
SEED_POINTER_CONTRACT = "vaaet-seed-bootstrap-pointer-v1"
HITL_CATALOG_CONTRACT = "vaaet-dataset-catalog-v1"
TRAINING_INPUT_LOCK_CONTRACT = "vaaet-training-input-lock-v1"
HITL_PACKAGE_CONTRACT = "vaaet-training-dataset-v1"
SEED_POINTER_FILE = "current.json"
HITL_CATALOG_FILE = "catalog.json"
HITL_PACKAGE_FILE = "vaaet-training-dataset-v1.zip"

_UUID_NAMESPACE = uuid.UUID("5ef88f18-4663-4c81-a6f9-5b40b256e083")


class DatasetArtifactAction(str, Enum):
    """Explicit action for an immutable, versioned dataset artifact."""

    REUSE_OR_CREATE = "reuse-or-create"
    CREATE_NEW_VERSION = "create-new-version"


class CatalogSelection(str, Enum):
    """Supported deterministic selections from a HITL package catalog."""

    ALL_ACTIVE = "all-active"


@dataclass(frozen=True)
class SeedArtifactConfig:
    store_root: Path
    action: DatasetArtifactAction = DatasetArtifactAction.REUSE_OR_CREATE
    update_reason: str | None = None
    git_commit: str = "unknown"
    vaaet_version: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_root", Path(self.store_root))
        object.__setattr__(self, "action", DatasetArtifactAction(self.action))
        if self.action is DatasetArtifactAction.CREATE_NEW_VERSION and not (
            self.update_reason and self.update_reason.strip()
        ):
            raise ValueError("Creating a new seed generation requires update_reason.")


@dataclass(frozen=True)
class SeedArtifactSnapshot:
    path: Path
    manifest: Mapping[str, Any]
    features: pd.DataFrame

    @property
    def descriptor(self) -> dict[str, object]:
        return {
            "contract": SEED_ARTIFACT_CONTRACT,
            "snapshot_id": str(self.manifest["snapshot_id"]),
            "generation": int(self.manifest["generation"]),
            "fingerprint": str(self.manifest["fingerprint"]),
            "sha256": str(self.manifest["package_sha256"]),
            "rows": int(len(self.features)),
            "filename": self.path.name,
        }


@dataclass(frozen=True)
class HitlCatalogSource:
    catalog_path: Path
    selection: CatalogSelection = CatalogSelection.ALL_ACTIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_path", Path(self.catalog_path))
        object.__setattr__(self, "selection", CatalogSelection(self.selection))


@dataclass(frozen=True)
class FinalizedReviewSession:
    package_id: str
    fingerprint: str
    package_sha256: str
    local_path: Path
    canonical_path: Path | None
    sync_status: str
    reviewed_rows: int
    pending_rows: int
    catalog_revision: int | None = None
    sync_error: str | None = None


@dataclass(frozen=True)
class TrainingInputLock:
    path: Path
    document: Mapping[str, Any]

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "contract": TRAINING_INPUT_LOCK_CONTRACT,
            "lock_id": str(self.document["lock_id"]),
            "fingerprint": str(self.document["fingerprint"]),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return value


def _atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("Catalog package path must be a non-empty relative path.")
    if "\\" in value or ":" in value:
        raise ValueError(f"Unsafe catalog package path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe catalog package path: {value}")
    return path


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "record_time" in result:
        result["record_time"] = normalize_timestamp_series(result["record_time"]).map(
            lambda value: value.isoformat()
        )
    if "reviewed_at" in result:
        reviewed = pd.to_datetime(result["reviewed_at"], utc=True, errors="coerce")
        result["reviewed_at"] = reviewed.map(
            lambda value: value.isoformat() if pd.notna(value) else ""
        )
    sort_columns = [
        column
        for column in ("clip_id", "record_time", "telemetry_feature_id", "prediction_id", "id")
        if column in result
    ]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable")
    return result.reset_index(drop=True)


def _frame_bytes(frame: pd.DataFrame, *, ignore_columns: Sequence[str] = ()) -> bytes:
    portable = _canonical_frame(frame).drop(columns=list(ignore_columns), errors="ignore")
    return portable.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _frames_fingerprint(frames: Mapping[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name in sorted(frames):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        ignore = ("reviewed_at",) if name == "validations" else ()
        digest.update(_frame_bytes(frames[name], ignore_columns=ignore))
        digest.update(b"\0")
    return digest.hexdigest()


def _prepare_seed_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"clip_id", "record_time", "traffic_state", *FEATURE_COLS}
    if frame.empty:
        raise ValueError("Seed snapshot cannot contain zero feature rows.")
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Seed snapshot is missing fields: {missing}")
    feature_order = [column for column in frame.columns if column in FEATURE_COLS]
    if feature_order != list(FEATURE_COLS):
        raise ValueError("Seed snapshot does not preserve the exact 19-feature order.")
    result = frame.copy()
    result["record_time"] = normalize_timestamp_series(result["record_time"])
    result["traffic_state"] = pd.to_numeric(result["traffic_state"], errors="raise").astype(int)
    if not result["traffic_state"].isin(MODEL_STATE_LABELS).all():
        raise ValueError("Seed snapshot may contain only stable proxy labels 0, 1, and 2.")
    if "is_human_validated" in result and result["is_human_validated"].fillna(False).any():
        raise ValueError("Seed snapshots cannot contain human-validated rows.")
    result["is_human_validated"] = False
    if "feature_schema_version" not in result:
        result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    versions = set(result["feature_schema_version"].dropna().astype(str))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"Seed feature schema is incompatible: {sorted(versions)}")
    for column in FEATURE_COLS:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result[FEATURE_COLS].isna().any().any():
        raise ValueError("Seed snapshot contains missing canonical feature values.")
    comparison = ["traffic_state", *FEATURE_COLS]
    for _, group in result.groupby(["clip_id", "record_time"], dropna=False):
        if len(group[comparison].drop_duplicates()) > 1:
            raise ValueError("Seed snapshot contains conflicting natural record keys.")
    metadata = [column for column in result if column not in FEATURE_COLS]
    return (
        result[[*metadata, *FEATURE_COLS]]
        .drop_duplicates(["clip_id", "record_time"], keep="last")
        .sort_values(["clip_id", "record_time"])
        .reset_index(drop=True)
    )


def _read_package_manifest(path: Path) -> dict[str, Any]:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "dataset-manifest.json" not in names or any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise ValueError("Dataset package has an unsafe or incomplete member list.")
            document = json.loads(archive.read("dataset-manifest.json").decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid dataset package: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Dataset package manifest must be a JSON object.")
    return document


class VersionedSeedStore:
    """Immutable seed snapshots with an atomically updated current pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshots = self.root / "snapshots"
        self.pointer_path = self.root / SEED_POINTER_FILE

    def _write_current_pointer(self, snapshot: SeedArtifactSnapshot) -> None:
        try:
            relative = snapshot.path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("Seed snapshot is outside its configured store.") from exc
        pointer = {
            "contract": SEED_POINTER_CONTRACT,
            "snapshot_contract": SEED_ARTIFACT_CONTRACT,
            "path": PurePosixPath(*relative.parts).as_posix(),
            **{
                key: value
                for key, value in snapshot.descriptor.items()
                if key != "contract"
            },
        }
        _atomic_json_write(self.pointer_path, pointer)

    def load_current(self) -> SeedArtifactSnapshot | None:
        if not self.pointer_path.is_file():
            return None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid seed pointer: {exc}") from exc
        if pointer.get("contract") != SEED_POINTER_CONTRACT:
            raise ValueError("Unsupported seed pointer contract.")
        if pointer.get("snapshot_contract") != SEED_ARTIFACT_CONTRACT:
            raise ValueError("Unsupported seed snapshot contract in pointer.")
        relative = _safe_relative_path(pointer.get("path"))
        snapshot = self.load_snapshot(self.root.joinpath(*relative.parts))
        for field, value in snapshot.descriptor.items():
            if field == "contract":
                continue
            if field in pointer and pointer[field] != value:
                raise ValueError(f"Seed pointer {field} does not match its snapshot.")
        return snapshot

    def load_snapshot(self, path: str | Path) -> SeedArtifactSnapshot:
        from vaaet_ml.data.ingestion import (
            DATASET_PACKAGE_CONTRACT,
            SEED_DATASET_PACKAGE_CONTRACT,
            load_dataset_package,
        )

        package = Path(path)
        frames = load_dataset_package(
            package,
            accepted_contracts=(SEED_DATASET_PACKAGE_CONTRACT, DATASET_PACKAGE_CONTRACT),
        )
        features = _prepare_seed_features(frames.get("features", pd.DataFrame()))
        manifest = _read_package_manifest(package)
        metadata = manifest.get("package_metadata", {})
        provenance = manifest.get("provenance", {})
        contract = manifest.get("contract_version")
        if contract == DATASET_PACKAGE_CONTRACT and not (
            provenance.get("training_mode") == "seed-bootstrap"
            and provenance.get("supervision") == "weak-proxy"
        ):
            raise ValueError("Legacy seed package lacks weak-proxy seed provenance.")
        if contract == SEED_DATASET_PACKAGE_CONTRACT:
            required = {
                "snapshot_id",
                "generation",
                "fingerprint",
                "created_at",
                "previous_snapshot_id",
                "update_reason",
            }
            if not isinstance(metadata, dict):
                raise ValueError("Seed package metadata must be a JSON object.")
            if missing := sorted(required - metadata.keys()):
                raise ValueError(f"Seed package metadata is incomplete: {missing}")
            try:
                uuid.UUID(str(metadata["snapshot_id"]))
            except ValueError as exc:
                raise ValueError("Seed snapshot_id must be a UUID.") from exc
            generation = metadata["generation"]
            if type(generation) is not int or generation < 1:
                raise ValueError("Seed generation must be a positive integer.")
        else:
            metadata = {
                "snapshot_id": str(uuid.uuid5(_UUID_NAMESPACE, _frames_fingerprint({"features": features}))),
                "generation": 0,
                "fingerprint": _frames_fingerprint({"features": features}),
                "created_at": "legacy",
                "previous_snapshot_id": None,
                "update_reason": "legacy import",
            }
        fingerprint = _frames_fingerprint({"features": features})
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("Seed snapshot content fingerprint mismatch.")
        normalized = {
            **metadata,
            "package_sha256": _sha256_file(package),
        }
        return SeedArtifactSnapshot(package.resolve(), normalized, features)

    def resolve(
        self, features: pd.DataFrame, config: SeedArtifactConfig
    ) -> SeedArtifactSnapshot:
        from vaaet_ml.data.ingestion import create_dataset_package

        prepared = _prepare_seed_features(features)
        fingerprint = _frames_fingerprint({"features": prepared})
        current = self.load_current()
        if current is not None and current.manifest["fingerprint"] == fingerprint:
            return current
        if current is not None and config.action is DatasetArtifactAction.REUSE_OR_CREATE:
            raise ValueError(
                "Processed seed data differs from the current immutable snapshot. "
                "Use CREATE_NEW_VERSION with an update reason."
            )
        if current is None and config.action is DatasetArtifactAction.CREATE_NEW_VERSION:
            raise ValueError("No seed snapshot exists; use REUSE_OR_CREATE for generation 0001.")

        generation = 1 if current is None else int(current.manifest["generation"]) + 1
        snapshot_id = str(uuid.uuid4())
        metadata = {
            "snapshot_id": snapshot_id,
            "generation": generation,
            "fingerprint": fingerprint,
            "created_at": _utc_now().isoformat(),
            "previous_snapshot_id": (
                str(current.manifest["snapshot_id"]) if current is not None else None
            ),
            "update_reason": (
                "initial processed seed snapshot"
                if current is None
                else str(config.update_reason).strip()
            ),
            "git_commit": config.git_commit,
            "vaaet_version": config.vaaet_version,
        }
        filename = f"vaaet-seed-bootstrap-v1-{generation:04d}-{fingerprint}.zip"
        final_path = self.snapshots / filename
        self.snapshots.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            try:
                snapshot = self.load_snapshot(final_path)
            except (OSError, ValueError) as exc:
                raise FileExistsError(
                    "Seed snapshot exists without a valid pointer and failed validation; "
                    f"the file was preserved for manual recovery: {final_path}"
                ) from exc
            if (
                snapshot.manifest["fingerprint"] != fingerprint
                or int(snapshot.manifest["generation"]) != generation
            ):
                raise FileExistsError(
                    "Seed snapshot path is occupied by an incompatible immutable artifact: "
                    f"{final_path}"
                )
            self._write_current_pointer(snapshot)
            return snapshot
        temporary = self.snapshots / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            create_dataset_package(
                temporary,
                features=prepared,
                provenance={
                    "training_mode": "seed-bootstrap",
                    "supervision": "weak-proxy",
                    "git_commit": config.git_commit,
                },
                contract_version=SEED_ARTIFACT_CONTRACT,
                package_metadata=metadata,
                overwrite=False,
            )
            candidate = self.load_snapshot(temporary)
            if (
                candidate.manifest["fingerprint"] != fingerprint
                or int(candidate.manifest["generation"]) != generation
            ):
                raise ValueError("Temporary seed snapshot validation returned different metadata.")
            os.replace(temporary, final_path)
            snapshot = self.load_snapshot(final_path)
            self._write_current_pointer(snapshot)
            return snapshot
        finally:
            temporary.unlink(missing_ok=True)

    def import_legacy(
        self, package_path: str | Path, config: SeedArtifactConfig
    ) -> SeedArtifactSnapshot:
        legacy = self.load_snapshot(package_path)
        return self.resolve(legacy.features, config)


class HitlReviewCatalog:
    """Atomic catalog of immutable, checksum-protected HITL review packages."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.path = Path(catalog_path)
        self.root = self.path.parent

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "contract": HITL_CATALOG_CONTRACT,
                "revision": 0,
                "updated_at": None,
                "entries": [],
            }
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid HITL catalog: {exc}") from exc
        self._validate(document)
        return document

    def _validate(self, document: object) -> None:
        if not isinstance(document, dict) or document.get("contract") != HITL_CATALOG_CONTRACT:
            raise ValueError("Unsupported HITL catalog contract.")
        if type(document.get("revision")) is not int or document["revision"] < 0:
            raise ValueError("HITL catalog revision must be a non-negative integer.")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError("HITL catalog entries must be a list.")
        package_ids: set[str] = set()
        paths: set[str] = set()
        for entry in entries:
            required = {
                "package_id",
                "path",
                "created_at",
                "pipeline_run_id",
                "sha256",
                "fingerprint",
                "clips",
                "rows",
                "human_support",
                "status",
                "feature_schema_version",
                "vaaet_version",
            }
            if not isinstance(entry, dict):
                raise ValueError("HITL catalog entries must be JSON objects.")
            if missing := sorted(required - entry.keys()):
                raise ValueError(f"HITL catalog entry is incomplete: {missing}")
            try:
                uuid.UUID(str(entry["package_id"]))
                uuid.UUID(str(entry["pipeline_run_id"]))
            except ValueError as exc:
                raise ValueError("Catalog package and pipeline run IDs must be UUIDs.") from exc
            relative = _safe_relative_path(entry["path"]).as_posix()
            if PurePosixPath(relative).name != HITL_PACKAGE_FILE:
                raise ValueError("Catalog entries must reference the contractual HITL filename.")
            if entry["package_id"] in package_ids or relative in paths:
                raise ValueError("HITL catalog contains duplicate IDs or paths.")
            package_ids.add(entry["package_id"])
            paths.add(relative)
            if not _is_sha256(entry["sha256"]) or not _is_sha256(entry["fingerprint"]):
                raise ValueError("HITL catalog checksums must be SHA-256.")
            if entry["status"] not in {"active", "quarantined"}:
                raise ValueError("HITL catalog status must be active or quarantined.")
            if entry["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
                raise ValueError("HITL catalog feature schema is incompatible.")
            try:
                created_at = datetime.fromisoformat(
                    str(entry["created_at"]).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError("HITL catalog created_at must be ISO-8601.") from exc
            if created_at.tzinfo is None:
                raise ValueError("HITL catalog created_at must include a timezone.")
            if type(entry["clips"]) is not int or entry["clips"] < 0:
                raise ValueError("HITL catalog clip count must be non-negative.")
            for field in ("rows", "human_support"):
                values = entry[field]
                if not isinstance(values, dict) or any(
                    type(value) is not int or value < 0 for value in values.values()
                ):
                    raise ValueError(f"HITL catalog {field} must contain non-negative counts.")
            if not isinstance(entry["vaaet_version"], str) or not entry["vaaet_version"]:
                raise ValueError("HITL catalog VAAET version must be non-empty.")

    def find(self, *, pipeline_run_id: str, fingerprint: str) -> dict[str, Any] | None:
        return next(
            (
                entry
                for entry in self.load()["entries"]
                if entry["pipeline_run_id"] == pipeline_run_id
                and entry["fingerprint"] == fingerprint
            ),
            None,
        )

    def register(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        document = self.load()
        existing = next(
            (
                item
                for item in document["entries"]
                if item["package_id"] == entry.get("package_id")
                or (
                    item["pipeline_run_id"] == entry.get("pipeline_run_id")
                    and item["fingerprint"] == entry.get("fingerprint")
                )
            ),
            None,
        )
        if existing is not None:
            if dict(existing) != dict(entry):
                raise ValueError("Catalog registration conflicts with an existing package.")
            return document
        updated = {
            **document,
            "revision": int(document["revision"]) + 1,
            "updated_at": _utc_now().isoformat(),
            "entries": [*document["entries"], dict(entry)],
        }
        self._validate(updated)
        _atomic_json_write(self.path, updated)
        return updated

    def selected_entries(
        self, selection: CatalogSelection = CatalogSelection.ALL_ACTIVE
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        selection = CatalogSelection(selection)
        document = self.load()
        if selection is CatalogSelection.ALL_ACTIVE:
            entries = [entry for entry in document["entries"] if entry["status"] == "active"]
        else:  # pragma: no cover - Enum guards this branch
            raise ValueError(f"Unsupported catalog selection: {selection}")
        entries.sort(key=lambda entry: (entry["created_at"], entry["package_id"]))
        return document, entries

    def set_status(self, package_id: str, status: str) -> dict[str, Any]:
        """Activate or quarantine a package without deleting its history."""
        if status not in {"active", "quarantined"}:
            raise ValueError("Catalog status must be active or quarantined.")
        package_id = str(uuid.UUID(str(package_id)))
        document = self.load()
        matches = [entry for entry in document["entries"] if entry["package_id"] == package_id]
        if not matches:
            raise KeyError(f"HITL catalog package not found: {package_id}")
        if matches[0]["status"] == status:
            return document
        entries = [
            {**entry, "status": status} if entry["package_id"] == package_id else entry
            for entry in document["entries"]
        ]
        updated = {
            **document,
            "revision": int(document["revision"]) + 1,
            "updated_at": _utc_now().isoformat(),
            "entries": entries,
        }
        self._validate(updated)
        _atomic_json_write(self.path, updated)
        return updated

    def package_path(self, entry: Mapping[str, Any]) -> Path:
        relative = _safe_relative_path(entry.get("path"))
        candidate = self.root.joinpath(*relative.parts).resolve()
        root = self.root.resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("Catalog package path escapes its root.")
        return candidate


def _stable_uuid(kind: str, *parts: object) -> str:
    material = "|".join([kind, *(str(part) for part in parts)])
    return str(uuid.uuid5(_UUID_NAMESPACE, material))


def _normalize_review_frames(
    classified: pd.DataFrame,
    validations: pd.DataFrame | Sequence[object],
    *,
    pipeline_run_id: str,
    model_version: str,
    finalized_at: datetime,
) -> dict[str, pd.DataFrame]:
    try:
        run_uuid = str(uuid.UUID(str(pipeline_run_id)))
    except ValueError as exc:
        raise ValueError("pipeline_run_id must be a UUID.") from exc
    if classified.empty:
        raise ValueError("A HITL review session requires classified feature rows.")
    required = {"clip_id", "record_time", *FEATURE_COLS}
    if missing := sorted(required - set(classified.columns)):
        raise ValueError(f"Classified review rows are missing fields: {missing}")
    features = classified.copy().reset_index(drop=True)
    features["record_time"] = normalize_timestamp_series(features["record_time"])
    existing_feature_ids = features.get("id", pd.Series(pd.NA, index=features.index)).astype("string")
    features["id"] = [
        str(value)
        if _valid_uuid(value)
        else _stable_uuid("feature", run_uuid, row.clip_id, row.record_time)
        for value, row in zip(existing_feature_ids, features.itertuples())
    ]
    features["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    feature_metadata = [
        column
        for column in features.columns
        if column not in FEATURE_COLS
        and column
        not in {
            "prediction_id",
            "traffic_state",
            "state_label",
            "confidence",
            "model_confidence",
            "model_traffic_state",
            "probability_margin",
            "decision_abstained",
            "measurement_reliable",
            "accident_rule_triggered",
            "accident_alert_started",
            "accident_evidence_score",
        }
    ]
    features = features[[*feature_metadata, *FEATURE_COLS]]

    source_prediction_ids = classified.get(
        "prediction_id", pd.Series(range(1, len(classified) + 1), index=classified.index)
    )
    prediction_ids = [
        value
        if _valid_uuid(value)
        else _stable_uuid("prediction", run_uuid, feature_id, model_version)
        for value, feature_id in zip(source_prediction_ids.astype(str), features["id"])
    ]
    id_map = {str(old): new for old, new in zip(source_prediction_ids, prediction_ids)}
    prediction_columns = [
        column
        for column in (
            "traffic_state",
            "state_label",
            "confidence",
            "model_traffic_state",
            "model_confidence",
            "probability_margin",
            "decision_abstained",
            "measurement_reliable",
            "accident_rule_triggered",
            "accident_alert_started",
            "accident_evidence_score",
        )
        if column in classified
    ]
    predictions = classified[prediction_columns].copy()
    predictions.insert(0, "model_version", model_version)
    predictions.insert(0, "telemetry_feature_id", features["id"].tolist())
    predictions.insert(0, "id", prediction_ids)
    predictions["pipeline_run_id"] = run_uuid

    if isinstance(validations, pd.DataFrame):
        validation_frame = validations.copy()
    else:
        records = [asdict(item) if is_dataclass(item) else dict(item) for item in validations]
        validation_frame = pd.DataFrame(records)
    if not validation_frame.empty:
        if "validation_id" in validation_frame and "id" not in validation_frame:
            validation_frame = validation_frame.rename(columns={"validation_id": "id"})
        if "prediction_id" not in validation_frame or "validated_state" not in validation_frame:
            raise ValueError("Review validations require prediction_id and validated_state.")
        validation_frame["prediction_id"] = validation_frame["prediction_id"].map(
            lambda value: id_map.get(str(value), str(value))
        )
        unknown = set(validation_frame["prediction_id"]) - set(prediction_ids)
        if unknown:
            raise ValueError(f"Validations reference predictions outside the session: {sorted(unknown)}")
        validation_frame["validated_state"] = pd.to_numeric(
            validation_frame["validated_state"], errors="raise"
        ).astype(int)
        if not validation_frame["validated_state"].isin((0, 1, 2, 3)).all():
            raise ValueError("Human validations must use public states 0 through 3.")
        supplied_ids = validation_frame.get("id", pd.Series(pd.NA, index=validation_frame.index))
        validation_frame["id"] = [
            str(value)
            if pd.notna(value) and _valid_uuid(value)
            else _stable_uuid(
                "validation",
                prediction_id,
                state,
                validation_frame.iloc[index].get("reviewer_id", "unknown"),
                validation_frame.iloc[index].get("notes", ""),
            )
            for index, (value, prediction_id, state) in enumerate(
                zip(
                    supplied_ids,
                    validation_frame["prediction_id"],
                    validation_frame["validated_state"],
                )
            )
        ]
        if "supersedes_validation_id" in validation_frame:
            validation_frame["supersedes_validation_id"] = validation_frame[
                "supersedes_validation_id"
            ].map(lambda value: str(value) if pd.notna(value) and value else pd.NA)
        else:
            validation_frame["supersedes_validation_id"] = pd.NA
        if "reviewed_at" not in validation_frame:
            validation_frame["reviewed_at"] = finalized_at.isoformat()
        validation_frame["pipeline_run_id"] = run_uuid
    else:
        validation_frame = pd.DataFrame(
            columns=[
                "id",
                "prediction_id",
                "validated_state",
                "reviewer_id",
                "reviewed_at",
                "notes",
                "review_source",
                "incident_context_reviewed",
                "supersedes_validation_id",
                "pipeline_run_id",
            ]
        )
    reviewed = set(validation_frame["prediction_id"].astype(str))
    predictions["review_status"] = predictions["id"].map(
        lambda value: "validated" if str(value) in reviewed else "unreviewed"
    )
    return {
        "features": features,
        "predictions": predictions,
        "validations": validation_frame,
    }


def _valid_uuid(value: object) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _human_support(validations: pd.DataFrame) -> dict[str, int]:
    if validations.empty:
        return {label: 0 for label in (*MODEL_STATE_LABELS.values(), "Accident")}
    labels = {**MODEL_STATE_LABELS, 3: "Accident"}
    return {
        label: int(validations["validated_state"].eq(state).sum())
        for state, label in labels.items()
    }


def finalize_review_session(
    *,
    classified: pd.DataFrame,
    validations: pd.DataFrame | Sequence[object],
    pipeline_run_id: str,
    model_version: str,
    git_commit: str,
    vaaet_version: str,
    local_root: str | Path,
    canonical_root: str | Path | None = None,
) -> FinalizedReviewSession:
    """Finalize one immutable review session and optionally register it in Drive."""
    from vaaet_ml.data.ingestion import create_dataset_package, load_dataset_package

    finalized_at = _utc_now()
    frames = _normalize_review_frames(
        classified,
        validations,
        pipeline_run_id=pipeline_run_id,
        model_version=model_version,
        finalized_at=finalized_at,
    )
    fingerprint = _frames_fingerprint(frames)
    package_id = _stable_uuid("hitl-package", pipeline_run_id, fingerprint)
    pending_root = Path(local_root) / "pending-sync"
    pending_dir = pending_root / f"{pipeline_run_id}_{fingerprint}"
    local_path = pending_dir / HITL_PACKAGE_FILE
    metadata = {
        "package_id": package_id,
        "pipeline_run_id": str(uuid.UUID(str(pipeline_run_id))),
        "finalized_at": finalized_at.isoformat(),
        "fingerprint": fingerprint,
        "model_version": model_version,
        "git_commit": git_commit,
        "vaaet_version": vaaet_version,
        "clips": sorted(frames["features"]["clip_id"].astype(str).unique().tolist()),
        "prediction_support": {
            str(state): int(count)
            for state, count in frames["predictions"]
            .get("traffic_state", pd.Series(dtype=int))
            .value_counts()
            .sort_index()
            .items()
        },
        "reviewed_rows": int(len(frames["validations"])),
        "pending_rows": int(frames["predictions"]["review_status"].eq("unreviewed").sum()),
        "human_support": _human_support(frames["validations"]),
    }
    if local_path.is_file():
        existing = _read_package_manifest(local_path).get("package_metadata", {})
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("Pending HITL package path contains different session content.")
        finalized_at = datetime.fromisoformat(str(existing["finalized_at"]).replace("Z", "+00:00"))
        metadata = existing
        load_dataset_package(local_path)
    else:
        pending_dir.mkdir(parents=True, exist_ok=True)
        create_dataset_package(
            local_path,
            features=frames["features"],
            predictions=frames["predictions"],
            validations=frames["validations"],
            provenance={"origin": "inference-human-review-session"},
            package_metadata=metadata,
            overwrite=False,
            include_empty_components=("validations",),
        )
        load_dataset_package(local_path)
    package_sha256 = _sha256_file(local_path)
    reviewed_rows = int(metadata["reviewed_rows"])
    pending_rows = int(metadata["pending_rows"])
    if canonical_root is None:
        return FinalizedReviewSession(
            package_id,
            fingerprint,
            package_sha256,
            local_path,
            None,
            "pending-sync",
            reviewed_rows,
            pending_rows,
        )

    root = Path(canonical_root)
    catalog = HitlReviewCatalog(root / HITL_CATALOG_FILE)
    existing = catalog.find(pipeline_run_id=str(pipeline_run_id), fingerprint=fingerprint)
    if existing is not None:
        canonical_path = catalog.package_path(existing)
        if not canonical_path.is_file() or _sha256_file(canonical_path) != existing["sha256"]:
            raise ValueError("Cataloged HITL package is missing or corrupted.")
        return FinalizedReviewSession(
            package_id,
            fingerprint,
            package_sha256,
            local_path,
            canonical_path,
            "synced",
            reviewed_rows,
            pending_rows,
            catalog.load()["revision"],
        )

    relative_dir = PurePosixPath(
        f"{finalized_at:%Y}",
        f"{finalized_at:%m}",
        f"{finalized_at:%d}",
        f"{finalized_at:%Y%m%dT%H%M%SZ}_{pipeline_run_id}_{fingerprint}",
    )
    relative_path = relative_dir / HITL_PACKAGE_FILE
    canonical_path = root.joinpath(*relative_path.parts)
    try:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        if canonical_path.exists():
            if _sha256_file(canonical_path) != package_sha256:
                raise ValueError("Immutable HITL destination already contains different data.")
        else:
            temporary = canonical_path.with_name(
                f".{canonical_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                shutil.copy2(local_path, temporary)
                if _sha256_file(temporary) != package_sha256:
                    raise ValueError("HITL package checksum changed during Drive synchronization.")
                os.replace(temporary, canonical_path)
            finally:
                temporary.unlink(missing_ok=True)
        clip_count = int(frames["features"]["clip_id"].nunique())
        entry = {
            "package_id": package_id,
            "path": relative_path.as_posix(),
            "created_at": str(metadata["finalized_at"]),
            "pipeline_run_id": str(uuid.UUID(str(pipeline_run_id))),
            "sha256": package_sha256,
            "fingerprint": fingerprint,
            "clips": clip_count,
            "rows": {
                "features": int(len(frames["features"])),
                "predictions": int(len(frames["predictions"])),
                "validations": reviewed_rows,
                "unreviewed": pending_rows,
            },
            "human_support": dict(metadata["human_support"]),
            "status": "active",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "vaaet_version": vaaet_version,
        }
        document = catalog.register(entry)
    except (OSError, ValueError) as exc:
        return FinalizedReviewSession(
            package_id,
            fingerprint,
            package_sha256,
            local_path,
            None,
            "pending-sync",
            reviewed_rows,
            pending_rows,
            sync_error=f"{type(exc).__name__}: {exc}",
        )
    return FinalizedReviewSession(
        package_id,
        fingerprint,
        package_sha256,
        local_path,
        canonical_path,
        "synced",
        reviewed_rows,
        pending_rows,
        int(document["revision"]),
    )


def import_legacy_hitl_package(
    package_path: str | Path,
    *,
    pipeline_run_id: str,
    git_commit: str,
    vaaet_version: str,
    local_root: str | Path,
    canonical_root: str | Path,
) -> FinalizedReviewSession:
    """Explicitly migrate one legacy mutable HITL ZIP into the immutable catalog."""
    from vaaet_ml.data.ingestion import load_dataset_package

    frames = load_dataset_package(package_path)
    features = frames.get("features", pd.DataFrame())
    predictions = frames.get("predictions", pd.DataFrame())
    validations = frames.get("validations", pd.DataFrame())
    if features.empty or predictions.empty:
        raise ValueError("Legacy HITL import requires feature and prediction tables.")
    required_predictions = {"id", "telemetry_feature_id", "model_version"}
    if missing := sorted(required_predictions - set(predictions.columns)):
        raise ValueError(f"Legacy HITL predictions are missing fields: {missing}")
    prediction_projection = predictions[["id", "telemetry_feature_id", "model_version"]].rename(
        columns={"id": "prediction_id", "model_version": "imported_model_version"}
    )
    classified = features.merge(
        prediction_projection,
        left_on="id",
        right_on="telemetry_feature_id",
        how="inner",
        validate="one_to_one",
    )
    if len(classified) != len(features):
        raise ValueError("Legacy HITL package does not relate every feature to one prediction.")
    model_versions = set(classified["imported_model_version"].dropna().astype(str))
    if len(model_versions) != 1:
        raise ValueError("Legacy HITL package must contain exactly one model version.")
    classified["model_version"] = next(iter(model_versions))
    classified = classified.drop(columns=["telemetry_feature_id", "imported_model_version"])
    return finalize_review_session(
        classified=classified,
        validations=validations,
        pipeline_run_id=pipeline_run_id,
        model_version=next(iter(model_versions)),
        git_commit=git_commit,
        vaaet_version=vaaet_version,
        local_root=local_root,
        canonical_root=canonical_root,
    )


def _deduplicate_uuid_rows(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "id" not in frame:
        raise ValueError(f"Catalog {name} rows require globally unique UUID id values.")
    if not frame["id"].map(_valid_uuid).all():
        raise ValueError(f"Catalog {name} contains non-UUID identifiers.")
    comparison = [column for column in frame.columns if not column.startswith("_catalog_")]
    for identifier, group in frame.groupby("id", dropna=False):
        normalized = group[comparison].fillna("<NULL>").astype(str)
        if len(normalized.drop_duplicates()) > 1:
            raise ValueError(f"Conflicting catalog {name} rows for UUID {identifier}.")
    return frame.drop_duplicates("id", keep="last").reset_index(drop=True)


def _resolve_validation_graph(validations: pd.DataFrame) -> pd.DataFrame:
    if validations.empty:
        return validations.copy()
    required = {"id", "prediction_id", "validated_state", "supersedes_validation_id"}
    if missing := sorted(required - set(validations.columns)):
        raise ValueError(f"Catalog validations are missing fields: {missing}")
    if not validations["prediction_id"].map(_valid_uuid).all():
        raise ValueError("Catalog validation prediction_id values must be UUIDs.")
    ids = set(validations["id"].astype(str))
    parents: dict[str, str | None] = {}
    children: dict[str, list[str]] = {identifier: [] for identifier in ids}
    prediction_by_id = dict(
        zip(validations["id"].astype(str), validations["prediction_id"].astype(str))
    )
    for row in validations.itertuples():
        identifier = str(row.id)
        raw_parent = getattr(row, "supersedes_validation_id")
        parent = None if pd.isna(raw_parent) or not str(raw_parent).strip() else str(raw_parent)
        if parent is not None:
            if parent not in ids:
                raise ValueError(f"Validation {identifier} supersedes an unknown validation {parent}.")
            if prediction_by_id[parent] != prediction_by_id[identifier]:
                raise ValueError("A validation cannot supersede a validation for another prediction.")
            children[parent].append(identifier)
        parents[identifier] = parent
    branches = {identifier: values for identifier, values in children.items() if len(values) > 1}
    if branches:
        raise ValueError(f"Human validation graph contains branches: {branches}")
    roots_by_prediction: dict[str, list[str]] = {}
    for identifier, parent in parents.items():
        if parent is None:
            roots_by_prediction.setdefault(prediction_by_id[identifier], []).append(identifier)
    if ambiguous := {
        prediction: roots for prediction, roots in roots_by_prediction.items() if len(roots) != 1
    }:
        raise ValueError(f"Human validation graph has conflicting roots: {ambiguous}")
    leaves: list[str] = []
    for prediction, roots in roots_by_prediction.items():
        current = roots[0]
        visited: set[str] = set()
        while True:
            if current in visited:
                raise ValueError(f"Human validation graph contains a cycle for {prediction}.")
            visited.add(current)
            if not children[current]:
                leaves.append(current)
                break
            current = children[current][0]
    return validations.loc[validations["id"].astype(str).isin(leaves)].copy()


def load_hitl_catalog_feedback(
    source: HitlCatalogSource,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and globally resolve effective human feedback from active packages."""
    from vaaet_ml.data.ingestion import load_dataset_package

    catalog = HitlReviewCatalog(source.catalog_path)
    document, entries = catalog.selected_entries(source.selection)
    if not entries:
        raise ValueError("The HITL catalog contains no active packages.")
    frames_by_kind: dict[str, list[pd.DataFrame]] = {
        "features": [],
        "predictions": [],
        "validations": [],
    }
    for entry in entries:
        package_path = catalog.package_path(entry)
        if not package_path.is_file():
            raise FileNotFoundError(f"Cataloged HITL package not found: {package_path}")
        if _sha256_file(package_path) != entry["sha256"]:
            raise ValueError(f"Cataloged HITL package checksum mismatch: {entry['package_id']}")
        package_frames = load_dataset_package(package_path)
        package_metadata = _read_package_manifest(package_path).get("package_metadata", {})
        if package_metadata.get("fingerprint") != entry["fingerprint"]:
            raise ValueError(f"Cataloged HITL package fingerprint mismatch: {entry['package_id']}")
        for kind in frames_by_kind:
            frame = package_frames.get(kind, pd.DataFrame()).copy()
            if not frame.empty:
                frame["_catalog_package_id"] = entry["package_id"]
                frames_by_kind[kind].append(frame)
    combined = {
        kind: (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        for kind, frames in frames_by_kind.items()
    }
    input_counts = {kind: int(len(frame)) for kind, frame in combined.items()}
    features = _deduplicate_uuid_rows(combined["features"], name="features")
    predictions = _deduplicate_uuid_rows(combined["predictions"], name="predictions")
    validations = _deduplicate_uuid_rows(combined["validations"], name="validations")
    duplicate_counts = {
        "features": input_counts["features"] - len(features),
        "predictions": input_counts["predictions"] - len(predictions),
        "validations": input_counts["validations"] - len(validations),
    }
    if features.empty or predictions.empty:
        raise ValueError("Active HITL packages contain no compatible features and predictions.")
    if validations.empty:
        feedback = pd.DataFrame()
    else:
        if not set(predictions["telemetry_feature_id"].astype(str)).issubset(
            set(features["id"].astype(str))
        ):
            raise ValueError("Catalog predictions reference missing feature UUIDs.")
        if not set(validations["prediction_id"].astype(str)).issubset(
            set(predictions["id"].astype(str))
        ):
            raise ValueError("Catalog validations reference missing prediction UUIDs.")
        latest = _resolve_validation_graph(validations)
        prediction_projection = predictions[["id", "telemetry_feature_id", "model_version"]]
        feedback = features.merge(
            prediction_projection,
            left_on="id",
            right_on="telemetry_feature_id",
            suffixes=("", "_prediction"),
        ).merge(latest, left_on="id_prediction", right_on="prediction_id")
        feedback["traffic_state"] = pd.to_numeric(
            feedback["validated_state"], errors="raise"
        ).astype(int)
        feedback["is_human_validated"] = True
        feedback["record_time"] = normalize_timestamp_series(feedback["record_time"])
    descriptor = {
        "contract": HITL_CATALOG_CONTRACT,
        "revision": int(document["revision"]),
        "catalog_sha256": _sha256_file(source.catalog_path),
        "package_ids": [entry["package_id"] for entry in entries],
        "package_fingerprints": [entry["fingerprint"] for entry in entries],
        "package_sha256": [entry["sha256"] for entry in entries],
        "resolved_validations": int(len(feedback)),
        "duplicate_rows_resolved": duplicate_counts,
        "corrections_resolved": int(
            validations.get("supersedes_validation_id", pd.Series(dtype=object))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
    }
    feedback.attrs["vaaet_provenance"] = descriptor
    return feedback, descriptor


def create_training_input_lock(
    output_root: str | Path,
    *,
    training_pipeline_run_id: str,
    training_mode: str,
    seed_snapshot: Mapping[str, object] | None,
    hitl_catalog: Mapping[str, object] | None,
    human_holdout: Mapping[str, object] | None,
    result_rows: Mapping[str, int],
    resolution: Mapping[str, int],
) -> TrainingInputLock:
    """Persist the exact immutable inputs selected for a model training run."""
    run_id = str(uuid.UUID(str(training_pipeline_run_id)))
    fingerprint_payload = _json_safe({
        "training_mode": training_mode,
        "seed_snapshot": dict(seed_snapshot) if seed_snapshot is not None else None,
        "hitl_catalog": dict(hitl_catalog) if hitl_catalog is not None else None,
        "human_holdout": dict(human_holdout) if human_holdout is not None else None,
        "result_rows": {key: int(value) for key, value in sorted(result_rows.items())},
        "resolution": {key: int(value) for key, value in sorted(resolution.items())},
    })
    canonical = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint = _sha256_bytes(canonical.encode("utf-8"))
    lock_id = _stable_uuid("training-input-lock", fingerprint)
    document = {
        "contract": TRAINING_INPUT_LOCK_CONTRACT,
        "lock_id": lock_id,
        "fingerprint": fingerprint,
        "created_at": _utc_now().isoformat(),
        "training_pipeline_run_id": run_id,
        **fingerprint_payload,
    }
    path = Path(output_root) / run_id / "training-input-lock.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("Training run already has a different input lock.")
        return TrainingInputLock(path.resolve(), existing)
    _atomic_json_write(path, document)
    return TrainingInputLock(path.resolve(), document)


__all__ = [
    "CatalogSelection",
    "DatasetArtifactAction",
    "FinalizedReviewSession",
    "HITL_CATALOG_CONTRACT",
    "HITL_CATALOG_FILE",
    "HitlCatalogSource",
    "HitlReviewCatalog",
    "SEED_ARTIFACT_CONTRACT",
    "SeedArtifactConfig",
    "SeedArtifactSnapshot",
    "TRAINING_INPUT_LOCK_CONTRACT",
    "TrainingInputLock",
    "VersionedSeedStore",
    "create_training_input_lock",
    "finalize_review_session",
    "import_legacy_hitl_package",
    "load_hitl_catalog_feedback",
]
