# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contrato, normalización y validación de holdouts humanos inmutables."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.continuity import normalize_continuity_frame
from vaaet.settings import FEATURE_COLS, MODEL_STATE_LABELS
from vaaet.timestamps import normalize_timestamp_series

from vaaet_ml.data.datasets import build_group_ids

HUMAN_HOLDOUT_CONTRACT = "vaaet-human-holdout-v2"
HUMAN_HOLDOUT_POINTER_CONTRACT = "vaaet-human-holdout-pointer-v2"
LEGACY_HUMAN_HOLDOUT_CONTRACT = "vaaet-human-holdout-v1"
HOLDOUT_MANIFEST_FILE = "holdout-manifest.json"
VALIDATION_RECORDS_FILE = "validation-records.csv"
TEST_RECORDS_FILE = "test-records.csv"
CURRENT_POINTER_FILE = "current.json"

PARTITION_FILES = {
    "validation": VALIDATION_RECORDS_FILE,
    "test": TEST_RECORDS_FILE,
}
IDENTITY_COLUMNS = ("clip_id", "continuity_id", "record_time", "feature_schema_version")
OPTIONAL_METADATA_COLUMNS = (
    "id",
    "source_record_id",
    "prediction_id",
    "model_version",
    "model_revision",
    "reviewer_id",
    "reviewed_at",
    "notes",
    "pipeline_run_id",
)
HOLDOUT_RECORD_COLUMNS = (
    "clip_id",
    "continuity_id",
    "record_time",
    "group_id",
    "feature_schema_version",
    "traffic_state",
    "state_label",
    "is_human_validated",
    *OPTIONAL_METADATA_COLUMNS,
    *FEATURE_COLS,
)
LEGACY_HOLDOUT_RECORD_COLUMNS = tuple(
    column for column in HOLDOUT_RECORD_COLUMNS if column not in {"continuity_id", "model_revision"}
)


class HumanHoldoutAction(str, Enum):
    """Acción explícita que selecciona el notebook para el ciclo del holdout."""

    REUSE_OR_CREATE = "reuse-or-create"
    CREATE_NEW_VERSION = "create-new-version"


@dataclass(frozen=True)
class HumanHoldoutConfig:
    """Configuración inmutable para resolver un snapshot de holdout humano."""

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
    """Registros exactos de validación y test representados por un snapshot."""

    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: Mapping[str, object]
    path: Path

    @property
    def reserved_groups(self) -> frozenset[str]:
        """Devuelve grupos excluidos de entrenamiento para evitar fuga."""

        groups = pd.concat(
            [self.validation["group_id"], self.test["group_id"]], ignore_index=True
        )
        return frozenset(groups.astype(str))

    @property
    def descriptor(self) -> dict[str, object]:
        """Expone la identidad mínima que puede acompañar a un bundle v3."""

        return {
            "contract": str(self.manifest["contract"]),
            "snapshot_id": str(self.manifest["snapshot_id"]),
            "generation": int(self.manifest["generation"]),
            "fingerprint": str(self.manifest["fingerprint"]),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
        }


def is_sha256(value: object) -> bool:
    """Valida la representación hexadecimal de un checksum SHA-256."""

    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def prepare_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza y valida filas humanas antes de asignarlas a un holdout."""

    _validate_holdout_source(frame)
    result = normalize_continuity_frame(frame)
    result["traffic_state"] = pd.to_numeric(
        result["traffic_state"], errors="raise"
    ).astype(int)
    _validate_holdout_labels(result)
    _validate_holdout_schema_and_features(result)
    _complete_holdout_metadata(result)
    _validate_holdout_natural_keys(result)
    return (
        result.loc[:, HOLDOUT_RECORD_COLUMNS]
        .sort_values(["group_id", "record_time", "clip_id"])
        .reset_index(drop=True)
    )


def _validate_holdout_source(frame: pd.DataFrame) -> None:
    required = {
        "clip_id", "record_time", "feature_schema_version", "traffic_state", "is_human_validated",
        *FEATURE_COLS,
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Human holdout source is missing fields: {missing}")
    if frame.empty:
        raise ValueError("Human holdout source contains zero records.")


def _validate_holdout_labels(frame: pd.DataFrame) -> None:
    if not frame["traffic_state"].isin(MODEL_STATE_LABELS).all():
        raise ValueError("Human holdout may contain only stable states 0, 1, and 2.")
    frame["is_human_validated"] = _validated_boolean(frame["is_human_validated"])
    if not frame["is_human_validated"].all():
        raise ValueError("Every holdout record must be human validated.")


def _validate_holdout_schema_and_features(frame: pd.DataFrame) -> None:
    versions = set(frame["feature_schema_version"].dropna().astype(str))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"Human holdout feature schema is incompatible: {sorted(versions)}")
    for column in FEATURE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.loc[:, FEATURE_COLS].isna().any().any():
        missing = frame.loc[:, FEATURE_COLS].columns[frame.loc[:, FEATURE_COLS].isna().any()].tolist()
        raise ValueError(f"Human holdout contains missing feature values: {missing}")


def _complete_holdout_metadata(frame: pd.DataFrame) -> None:
    expected_labels = frame["traffic_state"].map(MODEL_STATE_LABELS)
    if "state_label" in frame:
        supplied = frame["state_label"].astype("string")
        mismatch = supplied.notna() & supplied.ne(expected_labels.astype("string"))
        if mismatch.any():
            raise ValueError("Human holdout state code and label are inconsistent.")
    frame["state_label"] = expected_labels
    frame["group_id"] = build_group_ids(frame).astype(str)
    for column in OPTIONAL_METADATA_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA


def _validate_holdout_natural_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Human holdout contains duplicate natural record keys.")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serializa registros normalizados sin perder precisión ni zona horaria."""

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


def content_fingerprint(validation: pd.DataFrame, test: pd.DataFrame) -> str:
    """Calcula el fingerprint de ambas particiones y su rol contractual."""

    digest = hashlib.sha256()
    for name, frame in (("validation", validation), ("test", test)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(csv_bytes(frame))
    return digest.hexdigest()


def support(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Resume filas y grupos por estado estable para el manifiesto."""

    return {
        MODEL_STATE_LABELS[state]: {
            "rows": int(frame["traffic_state"].eq(state).sum()),
            "groups": int(frame.loc[frame["traffic_state"].eq(state), "group_id"].nunique()),
        }
        for state in MODEL_STATE_LABELS
    }


def validate_partition_contract(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Impide particiones vacías, fuga de grupos o estados sin soporte."""

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


def _validated_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "1": True, "false": False, "0": False})
    if mapped.isna().any():
        raise ValueError("Holdout is_human_validated contains invalid boolean values.")
    return mapped.astype(bool)


__all__ = [
    "CURRENT_POINTER_FILE",
    "HOLDOUT_MANIFEST_FILE",
    "HOLDOUT_RECORD_COLUMNS",
    "HUMAN_HOLDOUT_CONTRACT",
    "HUMAN_HOLDOUT_POINTER_CONTRACT",
    "LEGACY_HOLDOUT_RECORD_COLUMNS",
    "LEGACY_HUMAN_HOLDOUT_CONTRACT",
    "HumanHoldoutAction",
    "HumanHoldoutConfig",
    "HumanHoldoutSnapshot",
    "IDENTITY_COLUMNS",
    "PARTITION_FILES",
    "content_fingerprint",
    "csv_bytes",
    "is_sha256",
    "prepare_records",
    "support",
    "validate_partition_contract",
]
