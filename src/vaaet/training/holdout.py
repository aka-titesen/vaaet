"""Immutable, versioned human holdouts for recurrent HITL training."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.data.datasets import build_group_ids
from vaaet.data.timestamps import normalize_timestamp_series
from vaaet.settings import FEATURE_COLS, MODEL_STATE_LABELS

HUMAN_HOLDOUT_CONTRACT = "vaaet-human-holdout-v1"
HUMAN_HOLDOUT_POINTER_CONTRACT = "vaaet-human-holdout-pointer-v1"
HOLDOUT_MANIFEST_FILE = "holdout-manifest.json"
VALIDATION_RECORDS_FILE = "validation-records.csv"
TEST_RECORDS_FILE = "test-records.csv"
CURRENT_POINTER_FILE = "current.json"

_PARTITION_FILES = {
    "validation": VALIDATION_RECORDS_FILE,
    "test": TEST_RECORDS_FILE,
}
_IDENTITY_COLUMNS = ("clip_id", "record_time", "feature_schema_version")
_OPTIONAL_METADATA_COLUMNS = (
    "id",
    "source_record_id",
    "prediction_id",
    "model_version",
    "reviewer_id",
    "reviewed_at",
    "notes",
    "pipeline_run_id",
)
HOLDOUT_RECORD_COLUMNS = (
    "clip_id",
    "record_time",
    "group_id",
    "feature_schema_version",
    "traffic_state",
    "state_label",
    "is_human_validated",
    *_OPTIONAL_METADATA_COLUMNS,
    *FEATURE_COLS,
)


class HumanHoldoutAction(str, Enum):
    """Explicit lifecycle action selected by the training notebook."""

    REUSE_OR_CREATE = "reuse-or-create"
    CREATE_NEW_VERSION = "create-new-version"


@dataclass(frozen=True)
class HumanHoldoutConfig:
    """Configuration for resolving one immutable holdout snapshot."""

    store_root: Path
    action: HumanHoldoutAction = HumanHoldoutAction.REUSE_OR_CREATE
    update_reason: str | None = None
    validation_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    git_commit: str = "unknown"
    vaaet_version: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_root", Path(self.store_root))
        object.__setattr__(self, "action", HumanHoldoutAction(self.action))
        if not 0 < self.validation_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("Holdout validation_size and test_size must be between 0 and 1.")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("Holdout validation_size and test_size must sum to less than 1.")
        if self.action is HumanHoldoutAction.CREATE_NEW_VERSION and not (
            self.update_reason and self.update_reason.strip()
        ):
            raise ValueError("Creating a new holdout version requires update_reason.")


@dataclass(frozen=True)
class HumanHoldoutSnapshot:
    """Exact human validation and test records represented by one snapshot."""

    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: Mapping[str, Any]
    path: Path

    @property
    def reserved_groups(self) -> frozenset[str]:
        groups = pd.concat(
            [self.validation["group_id"], self.test["group_id"]], ignore_index=True
        )
        return frozenset(groups.astype(str))

    @property
    def descriptor(self) -> dict[str, object]:
        return {
            "contract": HUMAN_HOLDOUT_CONTRACT,
            "snapshot_id": str(self.manifest["snapshot_id"]),
            "generation": int(self.manifest["generation"]),
            "fingerprint": str(self.manifest["fingerprint"]),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validated_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "1": True, "false": False, "0": False})
    if mapped.isna().any():
        raise ValueError("Holdout is_human_validated contains invalid boolean values.")
    return mapped.astype(bool)


def _prepare_records(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "clip_id",
        "record_time",
        "feature_schema_version",
        "traffic_state",
        "is_human_validated",
        *FEATURE_COLS,
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Human holdout source is missing fields: {missing}")
    if frame.empty:
        raise ValueError("Human holdout source contains zero records.")

    result = frame.copy()
    result["record_time"] = normalize_timestamp_series(
        result["record_time"], field_name="holdout.record_time"
    )
    result["traffic_state"] = pd.to_numeric(
        result["traffic_state"], errors="raise"
    ).astype(int)
    if not result["traffic_state"].isin(MODEL_STATE_LABELS).all():
        raise ValueError("Human holdout may contain only stable states 0, 1, and 2.")
    result["is_human_validated"] = _validated_boolean(result["is_human_validated"])
    if not result["is_human_validated"].all():
        raise ValueError("Every holdout record must be human validated.")
    versions = set(result["feature_schema_version"].dropna().astype(str))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"Human holdout feature schema is incompatible: {sorted(versions)}")

    for column in FEATURE_COLS:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.loc[:, FEATURE_COLS].isna().any().any():
        missing_features = result.loc[:, FEATURE_COLS].columns[
            result.loc[:, FEATURE_COLS].isna().any()
        ].tolist()
        raise ValueError(f"Human holdout contains missing feature values: {missing_features}")

    expected_labels = result["traffic_state"].map(MODEL_STATE_LABELS)
    if "state_label" in result:
        supplied = result["state_label"].astype("string")
        mismatch = supplied.notna() & supplied.ne(expected_labels.astype("string"))
        if mismatch.any():
            raise ValueError("Human holdout state code and label are inconsistent.")
    result["state_label"] = expected_labels
    result["group_id"] = build_group_ids(result).astype(str)
    for column in _OPTIONAL_METADATA_COLUMNS:
        if column not in result:
            result[column] = pd.NA

    if result.duplicated(list(_IDENTITY_COLUMNS)).any():
        raise ValueError("Human holdout contains duplicate natural record keys.")
    return (
        result.loc[:, HOLDOUT_RECORD_COLUMNS]
        .sort_values(["group_id", "record_time", "clip_id"])
        .reset_index(drop=True)
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    portable = frame.copy()
    portable["record_time"] = normalize_timestamp_series(portable["record_time"]).map(
        lambda value: value.isoformat()
    )
    if "reviewed_at" in portable and portable["reviewed_at"].notna().any():
        reviewed = pd.to_datetime(portable["reviewed_at"], utc=True, errors="coerce")
        portable["reviewed_at"] = reviewed.map(
            lambda value: value.isoformat() if pd.notna(value) else ""
        )
    return portable.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _content_fingerprint(validation: pd.DataFrame, test: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for name, frame in (("validation", validation), ("test", test)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_csv_bytes(frame))
    return digest.hexdigest()


def _support(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        MODEL_STATE_LABELS[state]: {
            "rows": int(frame["traffic_state"].eq(state).sum()),
            "groups": int(frame.loc[frame["traffic_state"].eq(state), "group_id"].nunique()),
        }
        for state in MODEL_STATE_LABELS
    }


def _validate_partition_contract(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    if validation.empty or test.empty:
        raise ValueError("Frozen validation and test partitions must both be non-empty.")
    overlap = set(validation["group_id"]) & set(test["group_id"])
    if overlap:
        raise ValueError(f"Human holdout leaks groups across validation and test: {sorted(overlap)}")
    for name, frame in (("validation", validation), ("test", test)):
        missing_states = sorted(set(MODEL_STATE_LABELS) - set(frame["traffic_state"]))
        if missing_states:
            labels = [MODEL_STATE_LABELS[state] for state in missing_states]
            raise ValueError(f"Frozen {name} lacks stable state support: {labels}")


class FileSystemHoldoutStore:
    """Checksum-protected snapshot store usable through a mounted Google Drive."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def pointer_path(self) -> Path:
        return self.root / CURRENT_POINTER_FILE

    def load_current(self) -> HumanHoldoutSnapshot | None:
        if not self.pointer_path.is_file():
            return None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid human holdout pointer: {exc}") from exc
        if pointer.get("contract") != HUMAN_HOLDOUT_POINTER_CONTRACT:
            raise ValueError("Unsupported human holdout pointer contract.")
        filename = pointer.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Unsafe human holdout pointer filename.")
        snapshot = self.load_snapshot(self.root / filename)
        descriptor = snapshot.descriptor
        for field in ("snapshot_id", "generation", "fingerprint"):
            if pointer.get(field) != descriptor[field]:
                raise ValueError(f"Human holdout pointer {field} does not match its snapshot.")
        return snapshot

    def load_snapshot(self, path: str | Path) -> HumanHoldoutSnapshot:
        package = Path(path)
        if not package.is_file():
            raise FileNotFoundError(f"Human holdout snapshot not found: {package}")
        try:
            with zipfile.ZipFile(package) as archive:
                expected_members = {HOLDOUT_MANIFEST_FILE, *_PARTITION_FILES.values()}
                members = set(archive.namelist())
                if members != expected_members or any(Path(name).name != name for name in members):
                    raise ValueError("Human holdout ZIP contains unexpected or unsafe paths.")
                manifest = json.loads(archive.read(HOLDOUT_MANIFEST_FILE).decode("utf-8"))
                frames: dict[str, pd.DataFrame] = {}
                for partition, filename in _PARTITION_FILES.items():
                    payload = archive.read(filename)
                    metadata = manifest.get("files", {}).get(partition, {})
                    if metadata.get("filename") != filename:
                        raise ValueError(f"Unexpected holdout filename for {partition}.")
                    if metadata.get("sha256") != _sha256_bytes(payload):
                        raise ValueError(f"Checksum mismatch for holdout {partition}.")
                    frame = pd.read_csv(
                        io.BytesIO(payload),
                        float_precision="round_trip",
                    )
                    if list(frame.columns) != list(HOLDOUT_RECORD_COLUMNS):
                        raise ValueError(f"Column contract mismatch for holdout {partition}.")
                    if len(frame) != metadata.get("rows"):
                        raise ValueError(f"Row count mismatch for holdout {partition}.")
                    frames[partition] = _prepare_records(frame)
        except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid human holdout package: {exc}") from exc

        self._validate_manifest(manifest)
        validation = frames["validation"]
        test = frames["test"]
        _validate_partition_contract(validation, test)
        fingerprint = _content_fingerprint(validation, test)
        if fingerprint != manifest["fingerprint"]:
            raise ValueError("Human holdout content fingerprint mismatch.")
        return HumanHoldoutSnapshot(validation, test, manifest, package.resolve())

    def _validate_manifest(self, manifest: object) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("Human holdout manifest must be a JSON object.")
        required = {
            "contract",
            "snapshot_id",
            "generation",
            "created_at",
            "previous_snapshot_id",
            "update_reason",
            "git_commit",
            "vaaet_version",
            "feature_schema_version",
            "feature_columns",
            "selection",
            "files",
            "support",
            "fingerprint",
            "source_fingerprint",
            "source_groups",
        }
        if missing := sorted(required - manifest.keys()):
            raise ValueError(f"Human holdout manifest is missing fields: {missing}")
        if manifest["contract"] != HUMAN_HOLDOUT_CONTRACT:
            raise ValueError("Unsupported human holdout contract.")
        try:
            uuid.UUID(str(manifest["snapshot_id"]))
        except ValueError as exc:
            raise ValueError("Human holdout snapshot_id must be a UUID.") from exc
        if type(manifest["generation"]) is not int or manifest["generation"] < 1:
            raise ValueError("Human holdout generation must be a positive integer.")
        created_at = datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            raise ValueError("Human holdout created_at must include a timezone.")
        if manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Human holdout feature schema is incompatible.")
        if manifest["feature_columns"] != list(FEATURE_COLS):
            raise ValueError("Human holdout feature order is incompatible.")
        fingerprint = manifest["fingerprint"]
        if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
            raise ValueError("Human holdout fingerprint must be SHA-256.")
        source_fingerprint = manifest["source_fingerprint"]
        if not isinstance(source_fingerprint, str) or not _is_sha256(source_fingerprint):
            raise ValueError("Human holdout source_fingerprint must be SHA-256.")
        if not isinstance(manifest["source_groups"], list) or not all(
            isinstance(group, str) and group for group in manifest["source_groups"]
        ):
            raise ValueError("Human holdout source_groups must be non-empty strings.")

    def write_snapshot(
        self,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        *,
        generation: int,
        previous_snapshot_id: str | None,
        update_reason: str,
        source_fingerprint: str,
        source_groups: set[str],
        config: HumanHoldoutConfig,
    ) -> HumanHoldoutSnapshot:
        validation = _prepare_records(validation)
        test = _prepare_records(test)
        _validate_partition_contract(validation, test)
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_id = str(uuid.uuid4())
        fingerprint = _content_fingerprint(validation, test)
        filename = f"human-holdout-{generation:04d}-{snapshot_id}.zip"
        final_path = self.root / filename
        if final_path.exists():
            raise FileExistsError(f"Human holdout snapshot already exists: {final_path}")

        payloads = {
            "validation": _csv_bytes(validation),
            "test": _csv_bytes(test),
        }
        manifest = {
            "contract": HUMAN_HOLDOUT_CONTRACT,
            "snapshot_id": snapshot_id,
            "generation": int(generation),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "previous_snapshot_id": previous_snapshot_id,
            "update_reason": update_reason,
            "git_commit": config.git_commit,
            "vaaet_version": config.vaaet_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": list(FEATURE_COLS),
            "selection": {
                "algorithm": "constrained-grouped-temporal-v1",
                "validation_size": config.validation_size,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "timezone": "UTC",
            },
            "files": {
                partition: {
                    "filename": _PARTITION_FILES[partition],
                    "rows": int(len(validation if partition == "validation" else test)),
                    "sha256": _sha256_bytes(payload),
                    "columns": list(HOLDOUT_RECORD_COLUMNS),
                }
                for partition, payload in payloads.items()
            },
            "support": {
                "validation": _support(validation),
                "test": _support(test),
            },
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
            "source_groups": sorted(source_groups),
        }

        fd, temporary_name = tempfile.mkstemp(
            prefix=".human-holdout-", suffix=".zip", dir=self.root
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    HOLDOUT_MANIFEST_FILE,
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                for partition, payload in payloads.items():
                    archive.writestr(_PARTITION_FILES[partition], payload)
            os.replace(temporary_path, final_path)
            snapshot = self.load_snapshot(final_path)
            pointer = {
                "filename": filename,
                **snapshot.descriptor,
                "contract": HUMAN_HOLDOUT_POINTER_CONTRACT,
            }
            pointer_temp = self.root / f".{CURRENT_POINTER_FILE}.{uuid.uuid4().hex}.tmp"
            pointer_temp.write_text(
                json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(pointer_temp, self.pointer_path)
            return snapshot
        finally:
            temporary_path.unlink(missing_ok=True)


def _ensure_creation_support(frame: pd.DataFrame) -> None:
    groups = build_group_ids(frame)
    insufficient = {
        MODEL_STATE_LABELS[state]: int(groups.loc[frame["traffic_state"].eq(state)].nunique())
        for state in MODEL_STATE_LABELS
        if groups.loc[frame["traffic_state"].eq(state)].nunique() < 3
    }
    if insufficient:
        raise ValueError(
            "A frozen holdout requires at least three independent groups per stable state; "
            f"insufficient support: {insufficient}"
        )


def _initial_partitions(
    frame: pd.DataFrame, config: HumanHoldoutConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_creation_support(frame)
    group_states = {
        str(group): set(group_frame["traffic_state"].astype(int))
        for group, group_frame in frame.groupby("group_id", sort=False)
    }
    group_times = (
        frame.groupby("group_id")["record_time"].max().sort_values(ascending=False)
    )

    def select_groups(
        candidates: list[str], *, desired: int, remaining_per_state: int
    ) -> set[str]:
        selected: set[str] = set()

        def can_reserve(group: str) -> bool:
            remaining = set(candidates) - selected - {group}
            return all(
                sum(state in group_states[item] for item in remaining) >= remaining_per_state
                for state in MODEL_STATE_LABELS
            )

        for state in MODEL_STATE_LABELS:
            if any(state in group_states[group] for group in selected):
                continue
            match = next(
                (
                    group
                    for group in candidates
                    if group not in selected
                    and state in group_states[group]
                    and can_reserve(group)
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"Cannot reserve a leakage-safe holdout group for {MODEL_STATE_LABELS[state]}."
                )
            selected.add(match)
        for group in candidates:
            if len(selected) >= desired:
                break
            if group not in selected and can_reserve(group):
                selected.add(group)
        return selected

    ordered_groups = [str(group) for group in group_times.index]
    test_target = max(1, int(round(len(ordered_groups) * config.test_size)))
    test_groups = select_groups(
        ordered_groups, desired=test_target, remaining_per_state=2
    )
    remaining_groups = [group for group in ordered_groups if group not in test_groups]
    validation_candidates = remaining_groups.copy()
    random.Random(config.random_state).shuffle(validation_candidates)
    validation_target = max(1, int(round(len(ordered_groups) * config.validation_size)))
    validation_groups = select_groups(
        validation_candidates, desired=validation_target, remaining_per_state=1
    )
    validation = frame.loc[frame["group_id"].isin(validation_groups)].copy()
    test = frame.loc[frame["group_id"].isin(test_groups)].copy()
    _validate_partition_contract(validation, test)
    return validation, test


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(list(_IDENTITY_COLUMNS), drop=False).sort_index()


def _snapshot_conflicts(
    current: HumanHoldoutSnapshot, available: pd.DataFrame
) -> list[tuple[object, ...]]:
    frozen = _keyed(pd.concat([current.validation, current.test], ignore_index=True))
    candidate = _keyed(available)
    overlap = frozen.index.intersection(candidate.index)
    if overlap.empty:
        return []
    comparison_columns = ["traffic_state", *FEATURE_COLS]
    left = frozen.loc[overlap, comparison_columns].reset_index(drop=True)
    right = candidate.loc[overlap, comparison_columns].reset_index(drop=True)
    numeric_equal = left[FEATURE_COLS].astype(float).eq(right[FEATURE_COLS].astype(float))
    equal = left["traffic_state"].eq(right["traffic_state"]) & numeric_equal.all(axis=1)
    return [tuple(key) if isinstance(key, tuple) else (key,) for key in overlap[~equal]]


def _updated_partitions(
    current: HumanHoldoutSnapshot,
    available: pd.DataFrame,
    config: HumanHoldoutConfig,
    seen_groups: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = {
        **{group: "validation" for group in current.validation["group_id"].astype(str)},
        **{group: "test" for group in current.test["group_id"].astype(str)},
    }
    current_frames = {
        "validation": current.validation.copy(),
        "test": current.test.copy(),
    }
    updated: dict[str, pd.DataFrame] = {}
    for partition in ("validation", "test"):
        groups = {group for group, assigned in assignments.items() if assigned == partition}
        replacements = available.loc[available["group_id"].isin(groups)].copy()
        combined = pd.concat([current_frames[partition], replacements], ignore_index=True)
        updated[partition] = combined.drop_duplicates(
            list(_IDENTITY_COLUMNS), keep="last"
        )

    new_rows = available.loc[~available["group_id"].isin(seen_groups)].copy()
    if new_rows["group_id"].nunique() >= 3:
        group_times = (
            new_rows.groupby("group_id")["record_time"]
            .max()
            .sort_values(ascending=False)
        )
        ordered_groups = [str(group) for group in group_times.index]
        test_count = min(
            len(ordered_groups) - 2,
            max(1, int(round(len(ordered_groups) * config.test_size))),
        )
        test_groups = set(ordered_groups[:test_count])
        remaining = [group for group in ordered_groups if group not in test_groups]
        random.Random(config.random_state).shuffle(remaining)
        validation_count = min(
            len(remaining) - 1,
            max(1, int(round(len(ordered_groups) * config.validation_size))),
        )
        validation_groups = set(remaining[:validation_count])
        updated["validation"] = pd.concat(
            [
                updated["validation"],
                new_rows.loc[new_rows["group_id"].isin(validation_groups)],
            ],
            ignore_index=True,
        )
        updated["test"] = pd.concat(
            [updated["test"], new_rows.loc[new_rows["group_id"].isin(test_groups)]],
            ignore_index=True,
        )
    return _prepare_records(updated["validation"]), _prepare_records(updated["test"])


def resolve_human_holdout(
    validated_feedback: pd.DataFrame,
    config: HumanHoldoutConfig,
) -> HumanHoldoutSnapshot:
    """Create, reuse, or version a human-only validation/test benchmark."""
    available = _prepare_records(validated_feedback)
    source_fingerprint = _sha256_bytes(_csv_bytes(available))
    source_groups = set(available["group_id"].astype(str))
    store = FileSystemHoldoutStore(config.store_root)
    current = store.load_current()

    if current is None:
        if config.action is HumanHoldoutAction.CREATE_NEW_VERSION:
            raise ValueError(
                "No frozen holdout exists; use REUSE_OR_CREATE for the first snapshot."
            )
        validation, test = _initial_partitions(available, config)
        return store.write_snapshot(
            validation,
            test,
            generation=1,
            previous_snapshot_id=None,
            update_reason="initial frozen human holdout",
            source_fingerprint=source_fingerprint,
            source_groups=source_groups,
            config=config,
        )

    if config.action is HumanHoldoutAction.REUSE_OR_CREATE:
        conflicts = _snapshot_conflicts(current, available)
        if conflicts:
            sample = conflicts[:3]
            raise ValueError(
                "Current human validations contradict the frozen holdout. "
                "Create a new version with an update reason. "
                f"Conflicting keys (sample): {sample}"
            )
        return current

    if current.manifest.get("source_fingerprint") == source_fingerprint:
        return current
    seen_groups = set(current.manifest["source_groups"])
    validation, test = _updated_partitions(
        current, available, config, seen_groups=seen_groups
    )
    fingerprint = _content_fingerprint(validation, test)
    if fingerprint == current.manifest["fingerprint"]:
        return current
    return store.write_snapshot(
        validation,
        test,
        generation=int(current.manifest["generation"]) + 1,
        previous_snapshot_id=str(current.manifest["snapshot_id"]),
        update_reason=str(config.update_reason).strip(),
        source_fingerprint=source_fingerprint,
        source_groups=seen_groups | source_groups,
        config=config,
    )


def require_comparable_holdouts(
    left: Mapping[str, object] | None,
    right: Mapping[str, object] | None,
) -> None:
    """Reject model comparison across different human benchmark snapshots."""
    if not left or not right:
        raise ValueError("Both models must declare a frozen human holdout.")
    if left.get("fingerprint") != right.get("fingerprint"):
        raise ValueError("Models use different human holdout fingerprints.")


__all__ = [
    "CURRENT_POINTER_FILE",
    "FileSystemHoldoutStore",
    "HOLDOUT_RECORD_COLUMNS",
    "HUMAN_HOLDOUT_CONTRACT",
    "HumanHoldoutAction",
    "HumanHoldoutConfig",
    "HumanHoldoutSnapshot",
    "require_comparable_holdouts",
    "resolve_human_holdout",
]
