from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.data.dataset_artifacts import (
    CatalogSelection,
    DatasetArtifactAction,
    HitlCatalogSource,
    HitlReviewCatalog,
    SeedArtifactConfig,
    VersionedSeedStore,
    create_training_input_lock,
    finalize_review_session,
    import_legacy_hitl_package,
    load_hitl_catalog_feedback,
)
from vaaet_ml.data.ingestion import create_dataset_package
from vaaet_ml.data.review import HumanValidation
from vaaet_ml.settings import FEATURE_COLS


def _feature_values(value: float = 1.0) -> dict[str, float]:
    return {column: value + index / 100 for index, column in enumerate(FEATURE_COLS)}


def _seed_frame(*, speed: float = 10.0) -> pd.DataFrame:
    rows = []
    for index, state in enumerate((0, 1, 2)):
        row = {
            "clip_id": f"seed-{index}",
            "record_time": pd.Timestamp("2025-01-01T00:00:00Z")
            + pd.Timedelta(minutes=index),
            "traffic_state": state,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "data_origin": "real",
            "synthetic_scenario": "observed",
            **_feature_values(speed + index),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _classified_frame(*, clip_id: str = "clip-a") -> pd.DataFrame:
    rows = []
    for index, state in enumerate((0, 1)):
        rows.append(
            {
                "clip_id": clip_id,
                "record_time": pd.Timestamp("2026-08-10T18:00:00Z")
                + pd.Timedelta(minutes=index),
                "traffic_state": state,
                "state_label": ("Normal", "Reduced")[state],
                "confidence": 0.9,
                "model_version": "mlp-v2.1",
                **_feature_values(10.0 + index),
            }
        )
    return pd.DataFrame(rows)


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
        SeedArtifactConfig(
            store.root,
            action=DatasetArtifactAction.CREATE_NEW_VERSION,
        )


def test_seed_snapshot_preserves_high_precision_floats(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    row_count = 512
    frame = pd.DataFrame(
        {
            "clip_id": [f"seed-{index // 16}" for index in range(row_count)],
            "record_time": pd.date_range(
                "2025-01-01T00:00:00Z",
                periods=row_count,
                freq="min",
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
            snapshot.features[column].to_numpy(),
            expected[column].to_numpy(),
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


def test_review_finalization_is_idempotent_and_cataloged(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    classified = _classified_frame()
    decision = HumanValidation(
        prediction_id=1,
        validated_state=0,
        reviewer_id="facundo",
    )
    first = finalize_review_session(
        classified=classified,
        validations=[decision],
        pipeline_run_id=run_id,
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
        canonical_root=tmp_path / "drive" / "hitl-reviews",
    )
    second = finalize_review_session(
        classified=classified,
        validations=[decision],
        pipeline_run_id=run_id,
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
        canonical_root=tmp_path / "drive" / "hitl-reviews",
    )

    assert first.sync_status == "synced"
    assert second.canonical_path == first.canonical_path
    assert first.package_id == second.package_id
    catalog = HitlReviewCatalog(tmp_path / "drive" / "hitl-reviews" / "catalog.json")
    assert catalog.load()["revision"] == 1
    assert len(catalog.load()["entries"]) == 1


def test_hitl_package_preserves_high_precision_features(tmp_path: Path) -> None:
    classified = _classified_frame()
    rng = np.random.default_rng(7)
    for column in FEATURE_COLS:
        classified[column] = rng.normal(size=len(classified))
    decision = HumanValidation(
        prediction_id=1,
        validated_state=0,
        reviewer_id="facundo",
    )
    review_root = tmp_path / "drive" / "hitl-reviews"
    finalize_review_session(
        classified=classified,
        validations=[decision],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.1",
        local_root=tmp_path / "local",
        canonical_root=review_root,
    )

    feedback, _ = load_hitl_catalog_feedback(
        HitlCatalogSource(review_root / "catalog.json")
    )

    assert len(feedback) == 1
    expected = classified.iloc[0]
    for column in FEATURE_COLS:
        assert feedback.iloc[0][column] == expected[column]


def test_legacy_hitl_package_is_imported_explicitly(tmp_path: Path) -> None:
    classified = _classified_frame()
    classified["id"] = [10, 11]
    predictions = pd.DataFrame(
        [
            {"id": 20, "telemetry_feature_id": 10, "model_version": "mlp-v2.1"},
            {"id": 21, "telemetry_feature_id": 11, "model_version": "mlp-v2.1"},
        ]
    )
    validations = pd.DataFrame(
        [
            {
                "id": str(uuid.uuid4()),
                "prediction_id": 20,
                "validated_state": 0,
                "reviewer_id": "legacy-reviewer",
                "reviewed_at": "2026-08-10T20:00:00Z",
            }
        ]
    )
    legacy = create_dataset_package(
        tmp_path / "legacy-hitl.zip",
        features=classified,
        predictions=predictions,
        validations=validations,
    )
    result = import_legacy_hitl_package(
        legacy,
        pipeline_run_id=str(uuid.uuid4()),
        git_commit="legacy",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
        canonical_root=tmp_path / "reviews",
    )
    assert result.sync_status == "synced"
    feedback, _ = load_hitl_catalog_feedback(
        HitlCatalogSource(tmp_path / "reviews" / "catalog.json")
    )
    assert len(feedback) == 1
    assert feedback.iloc[0]["traffic_state"] == 0


def test_review_finalization_without_canonical_store_remains_pending(tmp_path: Path) -> None:
    result = finalize_review_session(
        classified=_classified_frame(),
        validations=[],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
    )
    assert result.sync_status == "pending-sync"
    assert result.reviewed_rows == 0
    assert result.pending_rows == 2
    assert result.local_path.is_file()


def test_review_finalization_preserves_pending_package_when_sync_fails(
    tmp_path: Path,
) -> None:
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("not a directory", encoding="utf-8")
    result = finalize_review_session(
        classified=_classified_frame(),
        validations=[],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
        canonical_root=blocked_root,
    )
    assert result.sync_status == "pending-sync"
    assert result.sync_error
    assert result.local_path.is_file()
    assert not (blocked_root / "catalog.json").exists()


def test_catalog_loader_excludes_unreviewed_predictions(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    reviewed = finalize_review_session(
        classified=_classified_frame(clip_id="reviewed"),
        validations=[HumanValidation(1, 1, "reviewer")],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local-a",
        canonical_root=root,
    )
    finalize_review_session(
        classified=_classified_frame(clip_id="unreviewed"),
        validations=[],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local-b",
        canonical_root=root,
    )

    feedback, descriptor = load_hitl_catalog_feedback(
        HitlCatalogSource(root / "catalog.json", CatalogSelection.ALL_ACTIVE)
    )
    assert len(feedback) == 1
    assert feedback.iloc[0]["traffic_state"] == 1
    assert descriptor["resolved_validations"] == 1
    assert reviewed.package_id in descriptor["package_ids"]


def test_catalog_quarantine_is_explicit_and_non_destructive(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    result = finalize_review_session(
        classified=_classified_frame(),
        validations=[HumanValidation(1, 0, "reviewer")],
        pipeline_run_id=str(uuid.uuid4()),
        model_version="mlp-v2.1",
        git_commit="abc",
        vaaet_version="4.5.0",
        local_root=tmp_path / "local",
        canonical_root=root,
    )
    catalog = HitlReviewCatalog(root / "catalog.json")
    catalog.set_status(result.package_id, "quarantined")
    _, active = catalog.selected_entries()
    assert active == []
    assert result.canonical_path.is_file()
    catalog.set_status(result.package_id, "active")
    _, active = catalog.selected_entries()
    assert [entry["package_id"] for entry in active] == [result.package_id]


def test_catalog_rejects_unsafe_relative_path(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "contract": "vaaet-dataset-catalog-v1",
                "revision": 1,
                "updated_at": "2026-08-10T00:00:00+00:00",
                "entries": [
                    {
                        "package_id": str(uuid.uuid4()),
                        "path": "../escape.zip",
                        "created_at": "2026-08-10T00:00:00+00:00",
                        "pipeline_run_id": str(uuid.uuid4()),
                        "sha256": "a" * 64,
                        "fingerprint": "b" * 64,
                        "clips": 1,
                        "rows": {},
                        "human_support": {},
                        "status": "active",
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "vaaet_version": "4.5.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe"):
        HitlReviewCatalog(catalog_path).load()


@pytest.mark.parametrize("unsafe_path", ["..\\escape.zip", "C:/escape.zip"])
def test_catalog_rejects_platform_specific_unsafe_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    catalog = HitlReviewCatalog(tmp_path / "catalog.json")
    entry = {
        "package_id": str(uuid.uuid4()),
        "path": unsafe_path,
        "created_at": "2026-08-10T00:00:00+00:00",
        "pipeline_run_id": str(uuid.uuid4()),
        "sha256": "a" * 64,
        "fingerprint": "b" * 64,
        "clips": 1,
        "rows": {},
        "human_support": {},
        "status": "active",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "vaaet_version": "4.5.0",
    }
    with pytest.raises(ValueError, match="Unsafe"):
        catalog.register(entry)


def test_catalog_rejects_cross_package_validation_branch(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    run_id = str(uuid.uuid4())
    feature_id = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    root_validation = str(uuid.uuid4())
    features = pd.DataFrame(
        [
            {
                "id": feature_id,
                "clip_id": "clip",
                "record_time": "2026-08-10T00:00:00+00:00",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                **_feature_values(),
            }
        ]
    )
    predictions = pd.DataFrame(
        [{"id": prediction_id, "telemetry_feature_id": feature_id, "model_version": "mlp-v2.1"}]
    )
    validations = pd.DataFrame(
        [
            {
                "id": root_validation,
                "prediction_id": prediction_id,
                "validated_state": 0,
                "reviewed_at": "2026-08-10T00:00:00Z",
                "supersedes_validation_id": pd.NA,
            },
            {
                "id": str(uuid.uuid4()),
                "prediction_id": prediction_id,
                "validated_state": 1,
                "reviewed_at": "2026-08-10T00:01:00Z",
                "supersedes_validation_id": root_validation,
            },
            {
                "id": str(uuid.uuid4()),
                "prediction_id": prediction_id,
                "validated_state": 2,
                "reviewed_at": "2026-08-10T00:02:00Z",
                "supersedes_validation_id": root_validation,
            },
        ]
    )
    package_dir = root / "2026" / "08" / "10" / "branch"
    package = package_dir / "vaaet-training-dataset-v1.zip"
    from vaaet_ml.data.dataset_artifacts import _frames_fingerprint, _sha256_file

    fingerprint = _frames_fingerprint(
        {"features": features, "predictions": predictions, "validations": validations}
    )
    create_dataset_package(
        package,
        features=features,
        predictions=predictions,
        validations=validations,
        package_metadata={"fingerprint": fingerprint},
    )

    entry = {
        "package_id": str(uuid.uuid4()),
        "path": package.relative_to(root).as_posix(),
        "created_at": "2026-08-10T00:00:00+00:00",
        "pipeline_run_id": run_id,
        "sha256": _sha256_file(package),
        "fingerprint": fingerprint,
        "clips": 1,
        "rows": {"features": 1, "predictions": 1, "validations": 3, "unreviewed": 0},
        "human_support": {},
        "status": "active",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "vaaet_version": "4.5.0",
    }
    HitlReviewCatalog(root / "catalog.json").register(entry)
    with pytest.raises(ValueError, match="branches"):
        load_hitl_catalog_feedback(HitlCatalogSource(root / "catalog.json"))


def test_catalog_resolves_valid_cross_package_correction_chain(tmp_path: Path) -> None:
    from vaaet_ml.data.dataset_artifacts import _frames_fingerprint, _sha256_file

    root = tmp_path / "reviews"
    feature_id = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    first_validation_id = str(uuid.uuid4())
    features = pd.DataFrame(
        [
            {
                "id": feature_id,
                "clip_id": "clip",
                "record_time": "2026-08-10T00:00:00+00:00",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                **_feature_values(),
            }
        ]
    )
    predictions = pd.DataFrame(
        [{"id": prediction_id, "telemetry_feature_id": feature_id, "model_version": "mlp-v2.1"}]
    )
    validation_frames = [
        pd.DataFrame(
            [
                {
                    "id": first_validation_id,
                    "prediction_id": prediction_id,
                    "validated_state": 0,
                    "reviewed_at": "2026-08-10T00:00:00Z",
                    "supersedes_validation_id": pd.NA,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "id": str(uuid.uuid4()),
                    "prediction_id": prediction_id,
                    "validated_state": 1,
                    "reviewed_at": "2026-08-11T00:00:00Z",
                    "supersedes_validation_id": first_validation_id,
                }
            ]
        ),
    ]
    catalog = HitlReviewCatalog(root / "catalog.json")
    for index, validations in enumerate(validation_frames, start=1):
        frames = {"features": features, "predictions": predictions, "validations": validations}
        fingerprint = _frames_fingerprint(frames)
        relative = Path("2026") / "08" / f"1{index}" / f"package-{index}" / "vaaet-training-dataset-v1.zip"
        package = root / relative
        create_dataset_package(
            package,
            features=features,
            predictions=predictions,
            validations=validations,
            package_metadata={"fingerprint": fingerprint},
        )
        catalog.register(
            {
                "package_id": str(uuid.uuid4()),
                "path": relative.as_posix(),
                "created_at": f"2026-08-1{index}T00:00:00+00:00",
                "pipeline_run_id": str(uuid.uuid4()),
                "sha256": _sha256_file(package),
                "fingerprint": fingerprint,
                "clips": 1,
                "rows": {"features": 1, "predictions": 1, "validations": 1, "unreviewed": 0},
                "human_support": {},
                "status": "active",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "vaaet_version": "4.5.0",
            }
        )
    feedback, descriptor = load_hitl_catalog_feedback(HitlCatalogSource(catalog.path))
    assert len(feedback) == 1
    assert feedback.iloc[0]["traffic_state"] == 1
    assert descriptor["resolved_validations"] == 1
    assert descriptor["corrections_resolved"] == 1
    assert descriptor["duplicate_rows_resolved"]["features"] == 1


def test_training_input_lock_is_reproducible_for_exact_inputs(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    kwargs = {
        "training_pipeline_run_id": run_id,
        "training_mode": "hitl-retraining",
        "seed_snapshot": {"fingerprint": "a" * 64},
        "hitl_catalog": {"revision": 2, "package_ids": [str(uuid.uuid4())]},
        "human_holdout": {"fingerprint": "b" * 64},
        "result_rows": {"train": 100, "validation": 20, "test": 20},
        "resolution": {"duplicates": 2, "corrections": 1},
    }
    first = create_training_input_lock(tmp_path, **kwargs)
    second = create_training_input_lock(tmp_path, **kwargs)
    assert first.descriptor == second.descriptor
    assert first.path == second.path
    assert first.path.is_file()
