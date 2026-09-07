# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Garantías atómicas de publicación del bundle candidato."""

from __future__ import annotations

from pathlib import Path

import pytest
from vaaet.artifacts import MANIFEST_FILE, REQUIRED_FILES, validate_manifest

from vaaet_ml.training.bundle_export import build_and_publish_bundle


class _Model:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, filepath: str | Path) -> None:
        Path(filepath).write_bytes(self.payload)


def _publish(destination: Path, payload: bytes) -> dict[str, object]:
    return build_and_publish_bundle(
        destination,
        model=_Model(payload),
        scaler={"scale": 1},
        label_mapping={0: "Normal", 1: "Reduced", 2: "Congested", 3: "Accident"},
        metrics={"direct_f1_macro": 0.5, "final_f1_macro": 0.5, "production_eligible": False},
        data_provenance={
            "origin": "test",
            "record_count": 3,
            "synthetic_data_included": False,
            "telemetry_v3_coverage": 0.0,
            "human_holdout": False,
            "production_eligible": False,
            "promotion_blockers": ["test bundle"],
        },
        training_lifecycle={
            "training_mode": "seed-bootstrap",
            "supervision": "weak-proxy",
            "deployment_stage": "pilot",
            "input_policy": "legacy-v1-bootstrap",
            "production_eligible": False,
        },
        decision_policy={"temperature": 1.0},
        human_holdout=None,
        training_input_lock=None,
    )


def test_bundle_is_published_only_after_full_validation(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"

    manifest = _publish(destination, b"first-model")

    assert validate_manifest(destination)["model_revision"] == manifest["model_revision"]
    assert all((destination / name).is_file() for name in (*REQUIRED_FILES, MANIFEST_FILE))


def test_failed_final_validation_restores_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "bundle"
    _publish(destination, b"first-model")
    original = {
        name: (destination / name).read_bytes() for name in (*REQUIRED_FILES, MANIFEST_FILE)
    }
    real_validate = validate_manifest
    calls = 0

    def fail_after_swap(path: str | Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated final validation failure")
        return real_validate(path)

    monkeypatch.setattr("vaaet_ml.training.bundle_export.validate_manifest", fail_after_swap)

    with pytest.raises(ValueError, match="simulated"):
        _publish(destination, b"second-model")

    assert {
        name: (destination / name).read_bytes() for name in (*REQUIRED_FILES, MANIFEST_FILE)
    } == original
    validate_manifest(destination)
