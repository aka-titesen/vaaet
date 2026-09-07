# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Almacenamiento ZIP atómico de snapshots humanos congelados."""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.settings import FEATURE_COLS

from vaaet_ml.data.artifact_serialization import sha256_bytes
from vaaet_ml.training.holdout_contract import (
    CURRENT_POINTER_FILE,
    HOLDOUT_MANIFEST_FILE,
    HOLDOUT_RECORD_COLUMNS,
    HUMAN_HOLDOUT_CONTRACT,
    HUMAN_HOLDOUT_POINTER_CONTRACT,
    LEGACY_HOLDOUT_RECORD_COLUMNS,
    LEGACY_HUMAN_HOLDOUT_CONTRACT,
    PARTITION_FILES,
    HumanHoldoutConfig,
    HumanHoldoutSnapshot,
    content_fingerprint,
    csv_bytes,
    is_sha256,
    prepare_records,
    support,
    validate_partition_contract,
)


class FileSystemHoldoutStore:
    """Store de snapshots con checksum apto para un Drive montado."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def pointer_path(self) -> Path:
        """Indica el puntero atómico al snapshot actualmente seleccionado."""

        return self.root / CURRENT_POINTER_FILE

    def load_current(self) -> HumanHoldoutSnapshot | None:
        """Carga el snapshot señalado por el puntero, o ``None`` si no existe."""

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
        for field in ("snapshot_id", "generation", "fingerprint"):
            if pointer.get(field) != snapshot.descriptor[field]:
                raise ValueError(f"Human holdout pointer {field} does not match its snapshot.")
        return snapshot

    def load_snapshot(self, path: str | Path) -> HumanHoldoutSnapshot:
        """Valida y carga un ZIP inmutable sin modificar sus archivos."""

        package = Path(path)
        if not package.is_file():
            raise FileNotFoundError(f"Human holdout snapshot not found: {package}")
        try:
            with zipfile.ZipFile(package) as archive:
                manifest, frames = self._read_archive(archive)
        except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid human holdout package: {exc}") from exc

        self._validate_manifest(manifest)
        validation = frames["validation"]
        test = frames["test"]
        validate_partition_contract(validation, test)
        if content_fingerprint(validation, test) != manifest["fingerprint"]:
            raise ValueError("Human holdout content fingerprint mismatch.")
        return HumanHoldoutSnapshot(validation, test, manifest, package.resolve())

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
        """Escribe un ZIP nuevo y actualiza el puntero sólo tras validarlo."""

        validation = prepare_records(validation)
        test = prepare_records(test)
        validate_partition_contract(validation, test)
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_id = str(uuid.uuid4())
        fingerprint = content_fingerprint(validation, test)
        filename = f"human-holdout-{generation:04d}-{snapshot_id}.zip"
        final_path = self.root / filename
        if final_path.exists():
            raise FileExistsError(f"Human holdout snapshot already exists: {final_path}")

        payloads = {
            "validation": csv_bytes(validation),
            "test": csv_bytes(test),
        }
        manifest = self._build_manifest(
            snapshot_id=snapshot_id,
            generation=generation,
            previous_snapshot_id=previous_snapshot_id,
            update_reason=update_reason,
            source_fingerprint=source_fingerprint,
            source_groups=source_groups,
            config=config,
            validation=validation,
            test=test,
            payloads=payloads,
            fingerprint=fingerprint,
        )
        temporary_path = self._write_zip(payloads, manifest)
        try:
            os.replace(temporary_path, final_path)
            snapshot = self.load_snapshot(final_path)
            self._write_current_pointer(filename, snapshot)
            return snapshot
        finally:
            temporary_path.unlink(missing_ok=True)

    def _read_archive(
        self, archive: zipfile.ZipFile
    ) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
        expected_members = {HOLDOUT_MANIFEST_FILE, *PARTITION_FILES.values()}
        members = set(archive.namelist())
        if members != expected_members or any(Path(name).name != name for name in members):
            raise ValueError("Human holdout ZIP contains unexpected or unsafe paths.")
        manifest = json.loads(archive.read(HOLDOUT_MANIFEST_FILE).decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Human holdout manifest must be a JSON object.")
        return manifest, {
            partition: self._read_partition(archive, manifest, partition, filename)
            for partition, filename in PARTITION_FILES.items()
        }

    def _read_partition(
        self,
        archive: zipfile.ZipFile,
        manifest: Mapping[str, object],
        partition: str,
        filename: str,
    ) -> pd.DataFrame:
        payload = archive.read(filename)
        files = manifest.get("files", {})
        metadata = files.get(partition, {}) if isinstance(files, Mapping) else {}
        if not isinstance(metadata, Mapping) or metadata.get("filename") != filename:
            raise ValueError(f"Unexpected holdout filename for {partition}.")
        if metadata.get("sha256") != sha256_bytes(payload):
            raise ValueError(f"Checksum mismatch for holdout {partition}.")
        frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
        columns = (
            LEGACY_HOLDOUT_RECORD_COLUMNS
            if manifest.get("contract") == LEGACY_HUMAN_HOLDOUT_CONTRACT
            else HOLDOUT_RECORD_COLUMNS
        )
        if list(frame.columns) != list(columns):
            raise ValueError(f"Column contract mismatch for holdout {partition}.")
        if len(frame) != metadata.get("rows"):
            raise ValueError(f"Row count mismatch for holdout {partition}.")
        if manifest.get("contract") == LEGACY_HUMAN_HOLDOUT_CONTRACT:
            frame["record_time"] = pd.to_datetime(frame["record_time"], utc=True)
            return frame.loc[:, columns].sort_values(
                ["group_id", "record_time", "clip_id"]
            ).reset_index(drop=True)
        return prepare_records(frame)

    def _validate_manifest(self, manifest: Mapping[str, object]) -> None:
        _validate_manifest_fields(manifest)
        _validate_manifest_identity(manifest)
        _validate_manifest_schema(manifest)
        _validate_manifest_checksums(manifest)
        _validate_manifest_source_groups(manifest)

    def _build_manifest(
        self,
        *,
        snapshot_id: str,
        generation: int,
        previous_snapshot_id: str | None,
        update_reason: str,
        source_fingerprint: str,
        source_groups: set[str],
        config: HumanHoldoutConfig,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        payloads: Mapping[str, bytes],
        fingerprint: str,
    ) -> dict[str, object]:
        frames = {"validation": validation, "test": test}
        return {
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
                    "filename": PARTITION_FILES[partition],
                    "rows": int(len(frames[partition])),
                    "sha256": sha256_bytes(payload),
                    "columns": list(HOLDOUT_RECORD_COLUMNS),
                }
                for partition, payload in payloads.items()
            },
            "support": {
                "validation": support(validation),
                "test": support(test),
            },
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
            "source_groups": sorted(source_groups),
        }

    def _write_zip(self, payloads: Mapping[str, bytes], manifest: Mapping[str, object]) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".human-holdout-", suffix=".zip", dir=self.root
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                HOLDOUT_MANIFEST_FILE,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
            for partition, payload in payloads.items():
                archive.writestr(PARTITION_FILES[partition], payload)
        return temporary_path

    def _write_current_pointer(self, filename: str, snapshot: HumanHoldoutSnapshot) -> None:
        pointer = {
            "filename": filename,
            **snapshot.descriptor,
            "contract": HUMAN_HOLDOUT_POINTER_CONTRACT,
        }
        temporary = self.root / f".{CURRENT_POINTER_FILE}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.pointer_path)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_manifest_fields(manifest: Mapping[str, object]) -> None:
    required = {
        "contract", "snapshot_id", "generation", "created_at", "previous_snapshot_id",
        "update_reason", "git_commit", "vaaet_version", "feature_schema_version", "feature_columns",
        "selection", "files", "support", "fingerprint", "source_fingerprint", "source_groups",
    }
    if missing := sorted(required - manifest.keys()):
        raise ValueError(f"Human holdout manifest is missing fields: {missing}")


def _validate_manifest_identity(manifest: Mapping[str, object]) -> None:
    if manifest["contract"] not in {
        HUMAN_HOLDOUT_CONTRACT,
        LEGACY_HUMAN_HOLDOUT_CONTRACT,
    }:
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


def _validate_manifest_schema(manifest: Mapping[str, object]) -> None:
    expected_schema = (
        "traffic-features-v2"
        if manifest["contract"] == LEGACY_HUMAN_HOLDOUT_CONTRACT
        else FEATURE_SCHEMA_VERSION
    )
    if manifest["feature_schema_version"] != expected_schema:
        raise ValueError("Human holdout feature schema is incompatible.")
    if manifest["feature_columns"] != list(FEATURE_COLS):
        raise ValueError("Human holdout feature order is incompatible.")


def _validate_manifest_checksums(manifest: Mapping[str, object]) -> None:
    for field in ("fingerprint", "source_fingerprint"):
        if not is_sha256(manifest[field]):
            raise ValueError(f"Human holdout {field} must be SHA-256.")


def _validate_manifest_source_groups(manifest: Mapping[str, object]) -> None:
    groups = manifest["source_groups"]
    if not isinstance(groups, list) or not all(isinstance(group, str) and group for group in groups):
        raise ValueError("Human holdout source_groups must be non-empty strings.")


__all__ = ["FileSystemHoldoutStore"]
