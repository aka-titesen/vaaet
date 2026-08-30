# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Validadores internos y tipados del manifiesto portable de bundles."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from vaaet.artifacts import (
    _SHA256_PATTERN,
    CONTRACT_VERSION,
    FEATURE_SCHEMA_VERSION,
    MANIFEST_FILE,
    MODEL_VERSION,
    REQUIRED_DEPENDENCIES,
    REQUIRED_FIELDS,
    REQUIRED_FILES,
    REQUIRED_POLICY_FIELDS,
    REQUIRED_PROVENANCE_FIELDS,
    TrafficBundleManifest,
    _sha256,
)
from vaaet.exceptions import ArtifactNotFoundError, ArtifactValidationError
from vaaet.lifecycle import ModelInputPolicy, TrainingMode
from vaaet.settings import FEATURE_COLS, MODEL_STATE_LABELS, STATE_LABELS


def validate_manifest(bundle_dir: str | Path) -> TrafficBundleManifest:
    """Valida compatibilidad e integridad antes de cargar un bundle v2."""

    directory = Path(bundle_dir).resolve()
    manifest = _load_manifest(directory)
    _validate_identity(manifest)
    lifecycle = _validate_lifecycle(manifest)
    _validate_policy(manifest)
    files, dependencies, metrics, provenance = _validate_sections(manifest)
    _validate_dependencies_and_metrics(dependencies, metrics)
    _validate_provenance(manifest, provenance)
    _validate_human_holdout(manifest, provenance)
    _validate_input_lock(manifest)
    _validate_eligibility(lifecycle, metrics, provenance)
    _validate_files(directory, files)
    return cast(TrafficBundleManifest, manifest)


def _load_manifest(directory: Path) -> dict[str, object]:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ArtifactNotFoundError(f"Missing artifact manifest: {MANIFEST_FILE}")
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Invalid artifact manifest: {exc}") from exc
    if not isinstance(raw_manifest, dict):
        raise ArtifactValidationError("Artifact manifest must be a JSON object.")
    return cast(dict[str, object], raw_manifest)


def _require_section(manifest: Mapping[str, object], field_name: str) -> dict[str, object]:
    value = manifest[field_name]
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"Manifest field '{field_name}' must be an object.")
    return cast(dict[str, object], value)


def _validate_identity(manifest: Mapping[str, object]) -> None:
    missing_fields = sorted(REQUIRED_FIELDS - manifest.keys())
    if missing_fields:
        raise ArtifactValidationError(f"Missing manifest fields: {', '.join(missing_fields)}")
    _validate_versions(manifest)
    _validate_generated_at(manifest["generated_at"])
    if not isinstance(manifest["git_commit"], str) or not manifest["git_commit"]:
        raise ArtifactValidationError("Manifest git_commit must be a non-empty string.")
    if manifest["feature_columns"] != list(FEATURE_COLS):
        raise ArtifactValidationError("Artifact feature schema does not match FEATURE_COLS.")
    if manifest["class_mapping"] != {str(key): value for key, value in STATE_LABELS.items()}:
        raise ArtifactValidationError("Artifact class mapping is incompatible with VAAET.")
    expected_outputs = {str(key): value for key, value in MODEL_STATE_LABELS.items()}
    if manifest["model_output_mapping"] != expected_outputs:
        raise ArtifactValidationError("Artifact MLP output mapping is incompatible with VAAET.")


def _validate_versions(manifest: Mapping[str, object]) -> None:
    if type(manifest["contract_version"]) is not int or manifest["contract_version"] != CONTRACT_VERSION:
        raise ArtifactValidationError("Unsupported artifact contract version.")
    if manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported feature schema version.")
    if manifest["model_version"] != MODEL_VERSION:
        raise ArtifactValidationError("Artifact model version is incompatible with VAAET.")


def _validate_generated_at(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError("Manifest generated_at must be a non-empty string.")
    try:
        generated_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError("Manifest generated_at must be a valid ISO-8601 timestamp.") from exc
    if generated_at.tzinfo is None:
        raise ArtifactValidationError("Manifest generated_at must include a timezone.")


def _validate_lifecycle(manifest: Mapping[str, object]) -> dict[str, object]:
    lifecycle = _require_section(manifest, "training_lifecycle")
    required_fields = {
        "training_mode",
        "supervision",
        "deployment_stage",
        "input_policy",
        "production_eligible",
    }
    if missing_fields := sorted(required_fields - lifecycle.keys()):
        raise ArtifactValidationError(
            f"Missing training lifecycle fields: {', '.join(missing_fields)}"
        )
    try:
        training_mode = TrainingMode(lifecycle["training_mode"])
        input_policy = ModelInputPolicy(lifecycle["input_policy"])
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("Unsupported training lifecycle mode or input policy.") from exc
    _validate_lifecycle_values(lifecycle, training_mode, input_policy)
    return lifecycle


def _validate_lifecycle_values(
    lifecycle: Mapping[str, object],
    training_mode: TrainingMode,
    input_policy: ModelInputPolicy,
) -> None:
    if lifecycle["deployment_stage"] not in {"pilot", "candidate", "production"}:
        raise ArtifactValidationError("Unsupported bundle deployment stage.")
    if not isinstance(lifecycle["supervision"], str) or not lifecycle["supervision"]:
        raise ArtifactValidationError("Training lifecycle supervision must be non-empty.")
    if type(lifecycle["production_eligible"]) is not bool:
        raise ArtifactValidationError("Training lifecycle eligibility must be boolean.")
    if training_mode is TrainingMode.SEED_BOOTSTRAP and (
        lifecycle["deployment_stage"] != "pilot"
        or lifecycle["production_eligible"]
        or lifecycle["supervision"] != "weak-proxy"
        or input_policy is not ModelInputPolicy.LEGACY_V1_BOOTSTRAP
    ):
        raise ArtifactValidationError(
            "Seed bootstrap bundles must remain legacy-policy weak-proxy pilots."
        )
    if training_mode is TrainingMode.HITL_RETRAINING and (
        lifecycle["deployment_stage"] not in {"candidate", "production"}
        or lifecycle["supervision"] != "human-validated-with-proxy-memory"
    ):
        raise ArtifactValidationError("HITL bundles must be human-validated candidates or production.")


def _validate_policy(manifest: Mapping[str, object]) -> dict[str, object]:
    policy = _require_section(manifest, "decision_policy")
    missing_fields = sorted(REQUIRED_POLICY_FIELDS - policy.keys())
    if missing_fields:
        raise ArtifactValidationError(
            f"Missing decision policy fields: {', '.join(missing_fields)}"
        )
    if policy["automatic_accident_state_allowed"] is not False:
        raise ArtifactValidationError("Bundle v2 must prohibit automatic Accident states.")
    if policy["human_confirmation_required_for_accident"] is not True:
        raise ArtifactValidationError("Bundle v2 must require human confirmation for Accident.")
    _validate_class_thresholds(policy["class_thresholds"])
    _validate_temperature(policy["temperature"])
    return policy


def _validate_class_thresholds(value: object) -> None:
    expected_keys = {str(key) for key in MODEL_STATE_LABELS}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ArtifactValidationError("Decision policy class thresholds are incompatible.")
    for threshold in value.values():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ArtifactValidationError("Decision thresholds must be numeric.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ArtifactValidationError("Decision thresholds must be between 0 and 1.")


def _validate_temperature(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError("Decision policy temperature must be numeric.")
    if not 0.0 < float(value) <= 10.0:
        raise ArtifactValidationError("Decision policy temperature is outside the supported range.")


def _validate_sections(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    return (
        _require_section(manifest, "files"),
        _require_section(manifest, "dependencies"),
        _require_section(manifest, "metrics"),
        _require_section(manifest, "data_provenance"),
    )


def _validate_dependencies_and_metrics(
    dependencies: Mapping[str, object], metrics: Mapping[str, object]
) -> None:
    for name in REQUIRED_DEPENDENCIES:
        version = dependencies.get(name)
        if not isinstance(version, str) or not version:
            raise ArtifactValidationError(f"Missing or invalid dependency version: {name}")
    f1_macro = metrics.get("f1_macro")
    if isinstance(f1_macro, bool) or not isinstance(f1_macro, (int, float)) or not 0 <= f1_macro <= 1:
        raise ArtifactValidationError("Manifest metrics.f1_macro must be a number between 0 and 1.")


def _validate_provenance(manifest: Mapping[str, object], provenance: Mapping[str, object]) -> None:
    missing_fields = sorted(REQUIRED_PROVENANCE_FIELDS - provenance.keys())
    if missing_fields:
        raise ArtifactValidationError(
            f"Missing data provenance fields: {', '.join(missing_fields)}"
        )
    if not isinstance(provenance["origin"], str) or not provenance["origin"]:
        raise ArtifactValidationError("Manifest data_provenance.origin must be a non-empty string.")
    if type(provenance["record_count"]) is not int or provenance["record_count"] < 0:
        raise ArtifactValidationError("Manifest data_provenance.record_count must be non-negative.")
    if type(provenance["synthetic_data_included"]) is not bool:
        raise ArtifactValidationError(
            "Manifest data_provenance.synthetic_data_included must be a boolean."
        )
    _validate_coverage(provenance["telemetry_v2_coverage"])
    for name in ("human_holdout", "production_eligible"):
        if type(provenance[name]) is not bool:
            raise ArtifactValidationError(f"Manifest data_provenance.{name} must be boolean.")
    if not isinstance(provenance["promotion_blockers"], list) or not all(
        isinstance(item, str) for item in provenance["promotion_blockers"]
    ):
        raise ArtifactValidationError("promotion_blockers must be a list of strings.")


def _validate_coverage(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError("telemetry_v2_coverage must be numeric.")
    if not 0.0 <= float(value) <= 1.0:
        raise ArtifactValidationError("telemetry_v2_coverage must be between 0 and 1.")


def _validate_human_holdout(manifest: Mapping[str, object], provenance: Mapping[str, object]) -> None:
    holdout = manifest.get("human_holdout")
    if not provenance["human_holdout"]:
        if holdout is not None:
            raise ArtifactValidationError(
                "A model without a frozen benchmark must not declare a human holdout descriptor."
            )
        return
    if not isinstance(holdout, dict):
        raise ArtifactValidationError(
            "A frozen human holdout requires its versioned snapshot descriptor."
        )
    descriptor = cast(dict[str, object], holdout)
    _require_descriptor_fields(
        descriptor,
        {"contract", "snapshot_id", "generation", "fingerprint", "validation_rows", "test_rows"},
        "human holdout descriptor",
    )
    if descriptor["contract"] != "vaaet-human-holdout-v1":
        raise ArtifactValidationError("Unsupported human holdout contract.")
    _validate_uuid(descriptor["snapshot_id"], "Human holdout snapshot_id")
    if type(descriptor["generation"]) is not int or descriptor["generation"] < 1:
        raise ArtifactValidationError("Human holdout generation must be positive.")
    _validate_fingerprint(descriptor["fingerprint"], "Human holdout fingerprint")
    for field_name in ("validation_rows", "test_rows"):
        if type(descriptor[field_name]) is not int or descriptor[field_name] < 1:
            raise ArtifactValidationError(
                f"Human holdout {field_name} must be a positive integer."
            )


def _validate_input_lock(manifest: Mapping[str, object]) -> None:
    input_lock = manifest.get("training_input_lock")
    if input_lock is None:
        return
    if not isinstance(input_lock, dict):
        raise ArtifactValidationError("training_input_lock must be an object or null.")
    descriptor = cast(dict[str, object], input_lock)
    _require_descriptor_fields(
        descriptor,
        {"contract", "lock_id", "fingerprint"},
        "training input lock",
    )
    if descriptor["contract"] != "vaaet-training-input-lock-v1":
        raise ArtifactValidationError("Unsupported training input lock contract.")
    _validate_uuid(descriptor["lock_id"], "Training input lock ID")
    _validate_fingerprint(descriptor["fingerprint"], "Training input lock fingerprint")


def _require_descriptor_fields(
    descriptor: Mapping[str, object],
    required_fields: set[str],
    descriptor_name: str,
) -> None:
    if missing_fields := sorted(required_fields - descriptor.keys()):
        raise ArtifactValidationError(
            f"Missing {descriptor_name} fields: {', '.join(missing_fields)}"
        )


def _validate_uuid(value: object, label: str) -> None:
    try:
        uuid.UUID(str(value))
    except ValueError as exc:
        raise ArtifactValidationError(f"{label} must be a UUID.") from exc


def _validate_fingerprint(value: object, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ArtifactValidationError(f"{label} must be SHA-256.")


def _validate_eligibility(
    lifecycle: Mapping[str, object],
    metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> None:
    metric_eligibility = metrics.get("production_eligible")
    if type(metric_eligibility) is not bool or metric_eligibility != provenance["production_eligible"]:
        raise ArtifactValidationError("Production eligibility must be explicit and consistent.")
    if lifecycle["production_eligible"] != metric_eligibility:
        raise ArtifactValidationError("Training lifecycle eligibility must match bundle metrics.")
    if lifecycle["deployment_stage"] == "production" and not metric_eligibility:
        raise ArtifactValidationError("Only eligible bundles may use the production stage.")


def _validate_files(directory: Path, files: Mapping[str, object]) -> None:
    for name in REQUIRED_FILES:
        path = directory / name
        file_entry = files.get(name)
        if not isinstance(file_entry, dict):
            raise ArtifactValidationError(f"Missing or invalid manifest file entry: {name}")
        expected = file_entry.get("sha256")
        if not path.is_file():
            raise ArtifactNotFoundError(f"Missing artifact file: {name}")
        if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
            raise ArtifactValidationError(f"Invalid SHA-256 entry for artifact file: {name}")
        if _sha256(path) != expected:
            raise ArtifactValidationError(f"Checksum mismatch for artifact file: {name}")
