from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaaet.artifacts import MANIFEST_FILE, REQUIRED_FILES, create_manifest, validate_manifest
from vaaet.exceptions import ArtifactNotFoundError, ArtifactValidationError


@pytest.fixture
def valid_bundle(tmp_path: Path) -> Path:
    for name in REQUIRED_FILES:
        (tmp_path / name).write_bytes(name.encode())
    create_manifest(
        tmp_path,
        metrics={"f1_macro": 0.85, "production_eligible": True},
        data_provenance={
            "origin": "test",
            "record_count": 1,
            "synthetic_data_included": False,
            "telemetry_v2_coverage": 1.0,
            "human_holdout": True,
            "production_eligible": True,
            "promotion_blockers": [],
        },
    )
    return tmp_path


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, payload: object) -> None:
    (bundle / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")


def test_valid_bundle(valid_bundle: Path) -> None:
    manifest = validate_manifest(valid_bundle)
    assert manifest["contract_version"] == 2
    assert manifest["model_output_mapping"] == {
        "0": "Normal",
        "1": "Reduced",
        "2": "Congested",
    }
    assert manifest["decision_policy"]["automatic_accident_state_allowed"] is False


@pytest.mark.parametrize("delta", [-1, 1])
def test_rejects_feature_schema_drift(valid_bundle: Path, delta: int) -> None:
    payload = _manifest(valid_bundle)
    features = list(payload["feature_columns"])
    payload["feature_columns"] = features[:-1] if delta < 0 else [*features, "unexpected"]
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="feature schema"):
        validate_manifest(valid_bundle)


def test_rejects_missing_manifest(valid_bundle: Path) -> None:
    (valid_bundle / MANIFEST_FILE).unlink()
    with pytest.raises(ArtifactNotFoundError, match="manifest"):
        validate_manifest(valid_bundle)


def test_rejects_missing_artifact(valid_bundle: Path) -> None:
    (valid_bundle / REQUIRED_FILES[0]).unlink()
    with pytest.raises(ArtifactNotFoundError, match="artifact file"):
        validate_manifest(valid_bundle)


def test_rejects_corrupt_json(valid_bundle: Path) -> None:
    (valid_bundle / MANIFEST_FILE).write_text("{broken", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="Invalid artifact manifest"):
        validate_manifest(valid_bundle)


def test_rejects_incompatible_mapping(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["class_mapping"] = {"0": "Unknown"}
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="class mapping"):
        validate_manifest(valid_bundle)


def test_rejects_four_mlp_outputs(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["model_output_mapping"]["3"] = "Accident"
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="MLP output mapping"):
        validate_manifest(valid_bundle)


def test_rejects_unsupported_contract(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["contract_version"] = 999
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="contract version"):
        validate_manifest(valid_bundle)


def test_rejects_policy_that_allows_automatic_accident(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["decision_policy"]["automatic_accident_state_allowed"] = True
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="prohibit automatic Accident"):
        validate_manifest(valid_bundle)


def test_rejects_invalid_calibration_temperature(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["decision_policy"]["temperature"] = 0
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="temperature"):
        validate_manifest(valid_bundle)


def test_rejects_missing_required_field(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    del payload["data_provenance"]
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="Missing manifest fields"):
        validate_manifest(valid_bundle)


def test_rejects_invalid_file_entry(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["files"][REQUIRED_FILES[0]] = "not-an-object"
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="manifest file entry"):
        validate_manifest(valid_bundle)


def test_rejects_invalid_provenance_types(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["data_provenance"]["synthetic_data_included"] = "false"
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="must be a boolean"):
        validate_manifest(valid_bundle)


def test_rejects_inconsistent_production_eligibility(valid_bundle: Path) -> None:
    payload = _manifest(valid_bundle)
    payload["metrics"]["production_eligible"] = False
    _write_manifest(valid_bundle, payload)
    with pytest.raises(ArtifactValidationError, match="eligibility"):
        validate_manifest(valid_bundle)


def test_rejects_changed_bundle_file(valid_bundle: Path) -> None:
    (valid_bundle / REQUIRED_FILES[0]).write_bytes(b"changed")
    with pytest.raises(ArtifactValidationError, match="Checksum"):
        validate_manifest(valid_bundle)
