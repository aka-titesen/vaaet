# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.data.dataset_artifacts import (
    CatalogSelection,
    HitlCatalogSource,
    HitlReviewCatalog,
    finalize_review_session,
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
