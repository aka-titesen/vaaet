# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Caracterización de snapshots semilla inmutables."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.data.package_codec import create_dataset_package
from vaaet_ml.data.seed_artifacts import (
    DatasetArtifactAction,
    SeedArtifactConfig,
    VersionedSeedStore,
    _prepare_seed_features,
)
from vaaet_ml.settings import FEATURE_COLS


def _feature_values(value: float = 1.0) -> dict[str, float]:
    return {column: value + index / 100 for index, column in enumerate(FEATURE_COLS)}


def _seed_frame(*, speed: float = 10.0) -> pd.DataFrame:
    rows = [
        {
            "clip_id": f"seed-{index}",
            "record_time": pd.Timestamp("2025-01-01T00:00:00Z")
            + pd.Timedelta(minutes=index),
            "traffic_state": state,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "data_origin": "real",
            "synthetic_scenario": "observed",
            **_feature_values(speed + index),
        }
        for index, state in enumerate((0, 1, 2))
    ]
    return pd.DataFrame(rows)


def _swap_last_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*frame.columns[:-2], FEATURE_COLS[-1], FEATURE_COLS[-2]]
    return frame[columns]


def test_seed_store_creates_reuses_and_versions_immutable_snapshots(tmp_path: Path) -> None:
    store = VersionedSeedStore(tmp_path / "seed")
    base_config = SeedArtifactConfig(store.root, git_commit="abc", vaaet_version="4.5.0")
    first = store.resolve(_seed_frame(), base_config)

    assert first.manifest["generation"] == 1
    assert first.path.is_file()
    assert store.resolve(_seed_frame(), base_config).path == first.path

    changed = _seed_frame(speed=20.0)
    with pytest.raises(ValueError, match="CREATE_NEW_VERSION"):
        store.resolve(changed, base_config)

    second = store.resolve(
        changed,
        SeedArtifactConfig(
            store.root,
            action=DatasetArtifactAction.CREATE_NEW_VERSION,
            update_reason="new calibrated seed",
            git_commit="def",
            vaaet_version="4.5.0",
        ),
    )
    assert second.manifest["generation"] == 2
    assert first.path.is_file()
    assert second.path != first.path
    assert store.load_current().path == second.path


def test_seed_store_rejects_changed_data_without_reason(tmp_path: Path) -> None:
    store = VersionedSeedStore(tmp_path / "seed")
    store.resolve(_seed_frame(), SeedArtifactConfig(store.root))
    with pytest.raises(ValueError, match="update_reason"):
        SeedArtifactConfig(store.root, action=DatasetArtifactAction.CREATE_NEW_VERSION)


def test_seed_snapshot_preserves_high_precision_floats(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    row_count = 512
    frame = pd.DataFrame(
        {
            "clip_id": [f"seed-{index // 16}" for index in range(row_count)],
            "record_time": pd.date_range(
                "2025-01-01T00:00:00Z", periods=row_count, freq="min"
            ),
            "traffic_state": np.arange(row_count) % 3,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "data_origin": "real",
            "synthetic_scenario": "observed",
        }
    )
    for column in FEATURE_COLS:
        frame[column] = rng.normal(size=row_count)

    store = VersionedSeedStore(tmp_path / "seed")
    snapshot = store.resolve(frame, SeedArtifactConfig(store.root))
    expected = frame.sort_values(["clip_id", "record_time"]).reset_index(drop=True)

    for column in FEATURE_COLS:
        np.testing.assert_array_equal(
            snapshot.features[column].to_numpy(), expected[column].to_numpy()
        )
    assert store.resolve(frame, SeedArtifactConfig(store.root)).path == snapshot.path


def test_seed_store_recovers_valid_snapshot_without_pointer(tmp_path: Path) -> None:
    frame = _seed_frame()
    store = VersionedSeedStore(tmp_path / "seed")
    created = store.resolve(frame, SeedArtifactConfig(store.root))
    package_bytes = created.path.read_bytes()
    store.pointer_path.unlink()

    recovered = store.resolve(frame, SeedArtifactConfig(store.root))

    assert recovered.path == created.path
    assert recovered.path.read_bytes() == package_bytes
    assert store.load_current().path == created.path


def test_seed_store_preserves_corrupt_snapshot_without_pointer(tmp_path: Path) -> None:
    frame = _seed_frame()
    store = VersionedSeedStore(tmp_path / "seed")
    created = store.resolve(frame, SeedArtifactConfig(store.root))
    store.pointer_path.unlink()
    corrupt_payload = b"not-a-valid-seed-package"
    created.path.write_bytes(corrupt_payload)

    with pytest.raises(FileExistsError, match="preserved for manual recovery"):
        store.resolve(frame, SeedArtifactConfig(store.root))

    assert created.path.read_bytes() == corrupt_payload
    assert not store.pointer_path.exists()


def test_seed_store_imports_explicit_legacy_weak_proxy_package(tmp_path: Path) -> None:
    legacy = create_dataset_package(
        tmp_path / "legacy-seed.zip",
        features=_seed_frame(),
        provenance={"training_mode": "seed-bootstrap", "supervision": "weak-proxy"},
    )
    store = VersionedSeedStore(tmp_path / "seed")
    snapshot = store.import_legacy(legacy, SeedArtifactConfig(store.root))
    assert snapshot.manifest["generation"] == 1
    assert snapshot.manifest["fingerprint"]
    assert legacy.is_file()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[0:0], "zero feature rows"),
        (lambda frame: frame.drop(columns=[FEATURE_COLS[0]]), "missing fields"),
        (_swap_last_feature_columns, "exact 19-feature order"),
        (lambda frame: frame.assign(traffic_state=3), "stable proxy labels"),
        (lambda frame: frame.assign(is_human_validated=True), "human-validated"),
        (lambda frame: frame.assign(feature_schema_version="unknown"), "schema is incompatible"),
        (lambda frame: frame.assign(**{FEATURE_COLS[0]: float("nan")}), "missing canonical"),
    ],
)
def test_seed_features_reject_invalid_contracts(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_seed_features(mutate(_seed_frame()))


def test_seed_features_reject_conflicting_natural_keys() -> None:
    frame = _seed_frame()
    conflicting = pd.concat([frame, frame.iloc[[0]].assign(**{FEATURE_COLS[0]: 99.0})])

    with pytest.raises(ValueError, match="conflicting natural record keys"):
        _prepare_seed_features(conflicting)


def test_seed_store_rejects_invalid_current_pointers(tmp_path: Path) -> None:
    store = VersionedSeedStore(tmp_path / "seed")
    snapshot = store.resolve(_seed_frame(), SeedArtifactConfig(store.root))
    pointer = store.pointer_path.read_text(encoding="utf-8")

    store.pointer_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid seed pointer"):
        store.load_current()

    store.pointer_path.write_text(pointer, encoding="utf-8")
    document = json.loads(pointer)
    document["generation"] = int(snapshot.manifest["generation"]) + 1
    store.pointer_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        store.load_current()


def test_seed_store_rejects_new_version_before_initial_snapshot(tmp_path: Path) -> None:
    store = VersionedSeedStore(tmp_path / "seed")
    config = SeedArtifactConfig(
        store.root,
        action=DatasetArtifactAction.CREATE_NEW_VERSION,
        update_reason="explicit",
    )

    with pytest.raises(ValueError, match="No seed snapshot exists"):
        store.resolve(_seed_frame(), config)


def test_seed_store_rejects_legacy_package_without_required_provenance(tmp_path: Path) -> None:
    legacy = create_dataset_package(tmp_path / "legacy.zip", features=_seed_frame())

    with pytest.raises(ValueError, match="weak-proxy seed provenance"):
        VersionedSeedStore(tmp_path / "seed").load_snapshot(legacy)
