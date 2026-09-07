# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Construcción y publicación atómica de bundles validados."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import joblib
from vaaet.artifacts import (
    LABEL_MAPPING_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    REQUIRED_FILES,
    SCALER_FILE,
    create_manifest,
    validate_manifest,
)


class SavableModel(Protocol):
    """Define el borde mínimo de guardado compatible con Keras."""

    def save(self, filepath: str | Path) -> None: ...


def build_and_publish_bundle(
    destination: str | Path,
    *,
    model: SavableModel,
    scaler: object,
    label_mapping: Mapping[int, str],
    metrics: Mapping[str, object],
    data_provenance: Mapping[str, object],
    training_lifecycle: Mapping[str, object],
    decision_policy: Mapping[str, object],
    human_holdout: Mapping[str, object] | None,
    training_input_lock: Mapping[str, object] | None,
) -> dict[str, object]:
    """Construye en staging y reemplaza la copia DVC sólo si valida completa."""

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        if (target / ".gitkeep").is_file():
            shutil.copy2(target / ".gitkeep", staging / ".gitkeep")
        model.save(staging / MODEL_FILE)
        joblib.dump(scaler, staging / SCALER_FILE)
        joblib.dump(dict(label_mapping), staging / LABEL_MAPPING_FILE)
        create_manifest(
            staging,
            metrics=metrics,
            data_provenance=data_provenance,
            training_lifecycle=training_lifecycle,
            decision_policy=decision_policy,
            human_holdout=human_holdout,
            training_input_lock=training_input_lock,
        )
        manifest = dict(validate_manifest(staging))
        _replace_validated_directory(staging, target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_bundle_copy(source: str | Path, destination: str | Path) -> dict[str, object]:
    """Copia a Drive mediante staging y valida antes y después del reemplazo."""

    source_path = Path(source).resolve()
    validate_manifest(source_path)
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for name in (*REQUIRED_FILES, MANIFEST_FILE):
            shutil.copy2(source_path / name, staging / name)
        manifest = dict(validate_manifest(staging))
        _replace_validated_directory(staging, target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _replace_validated_directory(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if target.exists():
            os.replace(target, backup)
            moved_existing = True
        os.replace(staging, target)
        validate_manifest(target)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if moved_existing and backup.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


__all__ = ["build_and_publish_bundle", "publish_bundle_copy"]
