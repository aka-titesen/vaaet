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
from vaaet.settings import FEATURE_COLS, MODEL_VERSION, STATE_LABELS

CONTRACT_VERSION = 1
FEATURE_SCHEMA_VERSION = "traffic-features-v1"
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
    "files",
    "dependencies",
    "metrics",
    "data_provenance",
}
REQUIRED_DEPENDENCIES = ("tensorflow", "scikit-learn", "joblib")
REQUIRED_PROVENANCE_FIELDS = {"origin", "record_count", "synthetic_data_included"}
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
) -> Path:
    """Create the serving manifest after a successful training export."""
    directory = Path(bundle_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ArtifactNotFoundError(f"Missing artifact files: {', '.join(missing)}")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(directory),
        "feature_columns": list(FEATURE_COLS),
        "class_mapping": {str(key): value for key, value in STATE_LABELS.items()},
        "files": {name: {"sha256": _sha256(directory / name)} for name in REQUIRED_FILES},
        "dependencies": {
            name: _installed_version(name) or "unknown" for name in REQUIRED_DEPENDENCIES
        },
        "metrics": dict(metrics),
        "data_provenance": dict(data_provenance),
    }
    output = directory / MANIFEST_FILE
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
