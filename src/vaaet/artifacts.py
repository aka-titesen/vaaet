"""Portable model-bundle contract shared with downstream serving systems."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vaaet.exceptions import ArtifactNotFoundError, ArtifactValidationError
from vaaet.settings import (
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_MIN_PROBABILITY_MARGIN,
    FEATURE_COLS,
    MODEL_STATE_LABELS,
    MODEL_VERSION,
    RECOVERY_PERSISTENCE_MINUTES,
    STATE_LABELS,
    WORSENING_PERSISTENCE_MINUTES,
)

CONTRACT_VERSION = 2
FEATURE_SCHEMA_VERSION = "traffic-features-v2"
MODEL_FILE = "traffic_classifier.keras"
SCALER_FILE = "feature_scaler.joblib"
LABEL_MAPPING_FILE = "label_mapping.joblib"
MANIFEST_FILE = "model-manifest.json"
REQUIRED_FILES = (MODEL_FILE, SCALER_FILE, LABEL_MAPPING_FILE)
REQUIRED_FIELDS = {
    "contract_version",
    "feature_schema_version",
    "model_version",
    "generated_at",
    "git_commit",
    "feature_columns",
    "class_mapping",
    "model_output_mapping",
    "decision_policy",
    "files",
    "dependencies",
    "metrics",
    "data_provenance",
}
REQUIRED_DEPENDENCIES = ("tensorflow", "scikit-learn", "joblib")
REQUIRED_PROVENANCE_FIELDS = {
    "origin",
    "record_count",
    "synthetic_data_included",
    "telemetry_v2_coverage",
    "human_holdout",
    "production_eligible",
    "promotion_blockers",
}
REQUIRED_POLICY_FIELDS = {
    "architecture",
    "class_thresholds",
    "minimum_probability_margin",
    "worsening_persistence_minutes",
    "recovery_persistence_minutes",
    "automatic_accident_state_allowed",
    "human_confirmation_required_for_accident",
    "temperature",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def create_manifest(
    bundle_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    decision_policy: Mapping[str, Any] | None = None,
) -> Path:
    """Create the serving manifest after a successful training export."""
    directory = Path(bundle_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ArtifactNotFoundError(f"Missing artifact files: {', '.join(missing)}")

    policy = {
        "architecture": "hierarchical-stable-flow-with-incident-candidate",
        "class_thresholds": {
            str(key): value for key, value in DEFAULT_CLASS_THRESHOLDS.items()
        },
        "minimum_probability_margin": DEFAULT_MIN_PROBABILITY_MARGIN,
        "worsening_persistence_minutes": WORSENING_PERSISTENCE_MINUTES,
        "recovery_persistence_minutes": RECOVERY_PERSISTENCE_MINUTES,
        "automatic_accident_state_allowed": False,
        "human_confirmation_required_for_accident": True,
        "temperature": 1.0,
    }
    if decision_policy:
        policy.update(dict(decision_policy))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(directory),
        "feature_columns": list(FEATURE_COLS),
        "class_mapping": {str(key): value for key, value in STATE_LABELS.items()},
        "model_output_mapping": {
            str(key): value for key, value in MODEL_STATE_LABELS.items()
        },
        "decision_policy": policy,
        "files": {name: {"sha256": _sha256(directory / name)} for name in REQUIRED_FILES},
        "dependencies": {
            name: _installed_version(name) or "unknown" for name in REQUIRED_DEPENDENCIES
        },
        "metrics": dict(metrics),
        "data_provenance": dict(data_provenance),
    }
    output = directory / MANIFEST_FILE
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_manifest(directory)
    return output


def validate_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate compatibility and integrity before loading a model bundle."""
    directory = Path(bundle_dir).resolve()
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ArtifactNotFoundError(f"Missing artifact manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Invalid artifact manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("Artifact manifest must be a JSON object.")

    missing_fields = sorted(REQUIRED_FIELDS - manifest.keys())
    if missing_fields:
        raise ArtifactValidationError(f"Missing manifest fields: {', '.join(missing_fields)}")
    if type(manifest["contract_version"]) is not int or manifest["contract_version"] != CONTRACT_VERSION:
        raise ArtifactValidationError("Unsupported artifact contract version.")
    if manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported feature schema version.")
    if manifest["model_version"] != MODEL_VERSION:
        raise ArtifactValidationError("Artifact model version is incompatible with VAAET.")
    if not isinstance(manifest["generated_at"], str) or not manifest["generated_at"]:
        raise ArtifactValidationError("Manifest generated_at must be a non-empty string.")
    try:
        generated_at = datetime.fromisoformat(manifest["generated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError("Manifest generated_at must be a valid ISO-8601 timestamp.") from exc
    if generated_at.tzinfo is None:
        raise ArtifactValidationError("Manifest generated_at must include a timezone.")
    if not isinstance(manifest["git_commit"], str) or not manifest["git_commit"]:
        raise ArtifactValidationError("Manifest git_commit must be a non-empty string.")
    if manifest["feature_columns"] != list(FEATURE_COLS):
        raise ArtifactValidationError("Artifact feature schema does not match FEATURE_COLS.")
    if manifest["class_mapping"] != {str(key): value for key, value in STATE_LABELS.items()}:
        raise ArtifactValidationError("Artifact class mapping is incompatible with VAAET.")
    expected_outputs = {str(key): value for key, value in MODEL_STATE_LABELS.items()}
    if manifest["model_output_mapping"] != expected_outputs:
        raise ArtifactValidationError("Artifact MLP output mapping is incompatible with VAAET.")
    policy = manifest["decision_policy"]
    if not isinstance(policy, dict):
        raise ArtifactValidationError("Manifest decision_policy must be an object.")
    missing_policy = sorted(REQUIRED_POLICY_FIELDS - policy.keys())
    if missing_policy:
        raise ArtifactValidationError(
            f"Missing decision policy fields: {', '.join(missing_policy)}"
        )
    if policy["automatic_accident_state_allowed"] is not False:
        raise ArtifactValidationError("Bundle v2 must prohibit automatic Accident states.")
    if policy["human_confirmation_required_for_accident"] is not True:
        raise ArtifactValidationError("Bundle v2 must require human confirmation for Accident.")
    expected_threshold_keys = {str(key) for key in MODEL_STATE_LABELS}
    thresholds = policy["class_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != expected_threshold_keys:
        raise ArtifactValidationError("Decision policy class thresholds are incompatible.")
    for threshold in thresholds.values():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ArtifactValidationError("Decision thresholds must be numeric.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ArtifactValidationError("Decision thresholds must be between 0 and 1.")
    temperature = policy["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ArtifactValidationError("Decision policy temperature must be numeric.")
    if not 0.0 < float(temperature) <= 10.0:
        raise ArtifactValidationError("Decision policy temperature is outside the supported range.")
    for field in ("files", "dependencies", "metrics", "data_provenance"):
        if not isinstance(manifest[field], dict):
            raise ArtifactValidationError(f"Manifest field '{field}' must be an object.")
    for name in REQUIRED_DEPENDENCIES:
        version = manifest["dependencies"].get(name)
        if not isinstance(version, str) or not version:
            raise ArtifactValidationError(f"Missing or invalid dependency version: {name}")
    f1_macro = manifest["metrics"].get("f1_macro")
    if isinstance(f1_macro, bool) or not isinstance(f1_macro, (int, float)) or not 0 <= f1_macro <= 1:
        raise ArtifactValidationError("Manifest metrics.f1_macro must be a number between 0 and 1.")
    missing_provenance = sorted(REQUIRED_PROVENANCE_FIELDS - manifest["data_provenance"].keys())
    if missing_provenance:
        raise ArtifactValidationError(
            f"Missing data provenance fields: {', '.join(missing_provenance)}"
        )
    provenance = manifest["data_provenance"]
    if not isinstance(provenance["origin"], str) or not provenance["origin"]:
        raise ArtifactValidationError("Manifest data_provenance.origin must be a non-empty string.")
    if type(provenance["record_count"]) is not int or provenance["record_count"] < 0:
        raise ArtifactValidationError("Manifest data_provenance.record_count must be non-negative.")
    if type(provenance["synthetic_data_included"]) is not bool:
        raise ArtifactValidationError(
            "Manifest data_provenance.synthetic_data_included must be a boolean."
        )
    if isinstance(provenance["telemetry_v2_coverage"], bool) or not isinstance(
        provenance["telemetry_v2_coverage"], (int, float)
    ):
        raise ArtifactValidationError("telemetry_v2_coverage must be numeric.")
    if not 0.0 <= float(provenance["telemetry_v2_coverage"]) <= 1.0:
        raise ArtifactValidationError("telemetry_v2_coverage must be between 0 and 1.")
    for name in ("human_holdout", "production_eligible"):
        if type(provenance[name]) is not bool:
            raise ArtifactValidationError(f"Manifest data_provenance.{name} must be boolean.")
    if not isinstance(provenance["promotion_blockers"], list) or not all(
        isinstance(item, str) for item in provenance["promotion_blockers"]
    ):
        raise ArtifactValidationError("promotion_blockers must be a list of strings.")
    metric_eligibility = manifest["metrics"].get("production_eligible")
    if type(metric_eligibility) is not bool or metric_eligibility != provenance["production_eligible"]:
        raise ArtifactValidationError("Production eligibility must be explicit and consistent.")

    for name in REQUIRED_FILES:
        path = directory / name
        file_entry = manifest["files"].get(name)
        if not isinstance(file_entry, dict):
            raise ArtifactValidationError(f"Missing or invalid manifest file entry: {name}")
        expected = file_entry.get("sha256")
        if not path.is_file():
            raise ArtifactNotFoundError(f"Missing artifact file: {path}")
        if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
            raise ArtifactValidationError(f"Invalid SHA-256 entry for artifact file: {name}")
        if _sha256(path) != expected:
            raise ArtifactValidationError(f"Checksum mismatch for artifact file: {name}")
    return manifest
