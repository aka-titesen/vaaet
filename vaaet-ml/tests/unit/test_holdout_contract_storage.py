# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Bordes del contrato y almacenamiento de holdouts humanos congelados."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.settings import FEATURE_COLS
from vaaet_ml.training.holdout import (
    FileSystemHoldoutStore,
    HumanHoldoutConfig,
    resolve_human_holdout,
)
from vaaet_ml.training.holdout_contract import (
    content_fingerprint,
    csv_bytes,
    prepare_records,
    validate_partition_contract,
)


def _record(*, state: int = 0, clip_id: str = "clip-a") -> dict[str, object]:
    return {
        "id": str(uuid.uuid4()),
        "clip_id": clip_id,
        "record_time": "2026-08-29T12:00:00Z",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "traffic_state": state,
        "is_human_validated": True,
        **{feature: float(index) for index, feature in enumerate(FEATURE_COLS)},
    }


def _feedback() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _record(state=index % 3, clip_id=f"clip-{index:02d}")
            | {"record_time": f"2026-08-{index + 1:02d}T12:00:00Z"}
            for index in range(15)
        ]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"validation_size": 0}, "between 0 and 1"),
        ({"validation_size": 0.6, "test_size": 0.4}, "sum to less"),
        ({"action": "create-new-version"}, "requires update_reason"),
    ],
)
def test_holdout_configuration_rejects_invalid_selection(
    kwargs: dict[str, object], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        HumanHoldoutConfig(tmp_path, **kwargs)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing fields"),
        (pd.DataFrame([_record()]).iloc[0:0], "zero records"),
        (pd.DataFrame([_record()]).assign(feature_schema_version="other"), "schema"),
        (pd.DataFrame([_record()]).assign(is_human_validated="maybe"), "invalid boolean"),
        (pd.DataFrame([_record()]).assign(state_label="wrong"), "inconsistent"),
    ],
)
def test_prepare_records_rejects_invalid_human_contracts(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_records(frame)


def test_record_serialization_and_partition_validation_are_deterministic() -> None:
    validation = prepare_records(pd.DataFrame([_record(state=0, clip_id="validation-a")]))
    test = prepare_records(pd.DataFrame([_record(state=0, clip_id="test-a")]))

    assert b"+00:00" in csv_bytes(validation.assign(reviewed_at="2026-08-29T12:00:00Z"))
    assert content_fingerprint(validation, test) == content_fingerprint(validation, test)
    with pytest.raises(ValueError, match="both be non-empty"):
        validate_partition_contract(validation.iloc[0:0], test)
    with pytest.raises(ValueError, match="leaks groups"):
        validate_partition_contract(validation, validation)
    with pytest.raises(ValueError, match="lacks stable state support"):
        validate_partition_contract(validation, test)


def test_holdout_store_rejects_corrupt_pointer_and_manifest(tmp_path: Path) -> None:
    snapshot = resolve_human_holdout(_feedback(), HumanHoldoutConfig(tmp_path))
    store = FileSystemHoldoutStore(tmp_path)
    manifest = dict(snapshot.manifest)

    (tmp_path / "current.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid human holdout pointer"):
        store.load_current()

    (tmp_path / "current.json").write_text(
        json.dumps({"contract": "unsupported"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsupported human holdout pointer"):
        store.load_current()

    for key, value, message in (
        ("contract", "unsupported", "Unsupported human holdout contract"),
        ("snapshot_id", "bad", "snapshot_id must be a UUID"),
        ("generation", 0, "generation must be a positive integer"),
        ("feature_columns", [], "feature order"),
        ("source_groups", [""], "source_groups"),
    ):
        invalid = {**manifest, key: value}
        with pytest.raises(ValueError, match=message):
            store._validate_manifest(invalid)


def test_holdout_store_rejects_missing_snapshot_and_pointer_mismatch(tmp_path: Path) -> None:
    store = FileSystemHoldoutStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="snapshot not found"):
        store.load_snapshot(tmp_path / "missing.zip")

    snapshot = resolve_human_holdout(_feedback(), HumanHoldoutConfig(tmp_path))
    pointer_path = tmp_path / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["fingerprint"] = "a" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        store.load_current()
    assert snapshot.path.is_file()
