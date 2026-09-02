# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from vaaet_ml.exceptions import RuntimeConfigurationError
from vaaet_ml.workflow_config import CollectionWorkflowConfig, TrainingWorkflowConfig
from vaaet_ml.workflow_presets import (
    CollectionPreset,
    EvaluationPreset,
    EvaluationPresetInputs,
    InferencePreset,
    TrainingPreset,
    collection_preset_config,
    evaluation_preset_config,
    inference_preset_config,
    render_workflow_summary,
    resolve_collection_config,
    resolve_evaluation_config,
    resolve_inference_config,
    resolve_training_config,
    training_preset_config,
)


def test_builtin_preset_ids_are_stable_and_versioned() -> None:
    presets = (
        *tuple(CollectionPreset),
        *tuple(TrainingPreset),
        *tuple(InferencePreset),
        *tuple(EvaluationPreset),
    )
    operational_ids = [preset.value for preset in presets if not preset.name.endswith("CUSTOM")]

    assert len(operational_ids) == len(set(operational_ids))
    assert all(preset_id.endswith("-v1") for preset_id in operational_ids)


def test_collection_presets_are_safe_and_immutable() -> None:
    local = collection_preset_config(CollectionPreset.LOCAL)
    postgres = collection_preset_config(CollectionPreset.POSTGRES)
    diagnostic = collection_preset_config(CollectionPreset.TRACKING_DIAGNOSTIC)

    assert local == CollectionWorkflowConfig(False, False)
    assert postgres.persist_to_database is True
    assert diagnostic.hud_debug is True
    assert local.download_outputs is False
    with pytest.raises(FrozenInstanceError):
        local.hud_debug = True  # type: ignore[misc]


def test_training_presets_preserve_governed_defaults() -> None:
    seed_upload = training_preset_config(TrainingPreset.SEED_UPLOAD)
    hitl_frozen = training_preset_config(TrainingPreset.HITL_FROZEN_HOLDOUT)

    assert seed_upload.training_mode == "seed_bootstrap"
    assert seed_upload.enable_data_upload is True
    assert seed_upload.enable_postgres_ingestion is False
    assert seed_upload.copy_bundle_to_drive is False
    assert seed_upload.run_grouped_cross_validation is False
    assert hitl_frozen.training_mode == "hitl_retraining"
    assert hitl_frozen.human_holdout_frozen is True


def test_inference_presets_keep_remote_effects_explicit() -> None:
    offline = inference_preset_config(InferencePreset.PILOT_OFFLINE)
    persisted_hitl = inference_preset_config(InferencePreset.PERSISTED_HITL)
    experimental = inference_preset_config(InferencePreset.EXPERIMENTAL_OFFLINE)

    assert offline.persist_to_database is False
    assert offline.enable_human_review is False
    assert offline.download_annotated_video is False
    assert persisted_hitl.persist_to_database is True
    assert persisted_hitl.enable_human_review is True
    assert experimental.allow_experimental_bundle is True
    assert experimental.persist_to_database is False


@pytest.mark.parametrize(
    ("resolver", "preset"),
    (
        (resolve_collection_config, CollectionPreset.CUSTOM),
        (resolve_training_config, TrainingPreset.CUSTOM),
        (resolve_inference_config, InferencePreset.CUSTOM),
    ),
)
def test_custom_presets_require_a_typed_configuration(
    resolver: object, preset: object
) -> None:
    with pytest.raises(RuntimeConfigurationError, match="CUSTOM requires"):
        resolver(preset)  # type: ignore[operator]


def test_custom_configuration_is_rejected_for_a_named_preset() -> None:
    custom = replace(collection_preset_config(CollectionPreset.LOCAL), hud_debug=True)

    with pytest.raises(RuntimeConfigurationError, match="valid only"):
        resolve_collection_config(CollectionPreset.LOCAL, custom_config=custom)


def test_dataclass_replace_revalidates_custom_configuration() -> None:
    base = training_preset_config(TrainingPreset.HITL_FROZEN_HOLDOUT)

    with pytest.raises(RuntimeConfigurationError, match="human_holdout_update_reason"):
        replace(
            base,
            human_holdout_action="create_new_version",
            human_holdout_update_reason=None,
        )


def test_evaluation_presets_require_only_their_own_inputs() -> None:
    comparison = evaluation_preset_config(
        EvaluationPreset.COMPARE_MODELS,
        inputs=EvaluationPresetInputs(
            champion_bundle_dir="champion",
            challenger_bundle_dir="challenger",
            holdout_snapshot_path="human-holdout-0001.zip",
        ),
    )
    postgres_drift = evaluation_preset_config(
        EvaluationPreset.POSTGRES_DRIFT,
        inputs=EvaluationPresetInputs(
            reference_feature_cohort_path="reference.csv",
            postgres_start_utc="2026-01-01T00:00:00Z",
            postgres_end_utc="2026-02-01T00:00:00Z",
            postgres_pipeline_run_ids=("run-1",),
            postgres_clip_ids=("clip-1",),
            drift_plot_features=("avg_speed",),
        ),
    )

    assert comparison.run_model_evaluation is True
    assert comparison.run_drift_analysis is False
    assert postgres_drift.use_postgres_operational is True
    assert postgres_drift.postgres_pipeline_run_ids == ("run-1",)
    assert postgres_drift.postgres_clip_ids == ("clip-1",)
    assert postgres_drift.drift_plot_features == ("avg_speed",)


def test_evaluation_preset_rejects_incompatible_inputs() -> None:
    with pytest.raises(RuntimeConfigurationError, match="another evaluation flow"):
        evaluation_preset_config(
            EvaluationPreset.FILE_DRIFT,
            inputs=EvaluationPresetInputs(
                reference_feature_cohort_path="reference.csv",
                operational_feature_cohort_path="operational.csv",
                postgres_start_utc="2026-01-01T00:00:00Z",
            ),
        )


def test_custom_evaluation_rejects_preset_inputs() -> None:
    custom = evaluation_preset_config(EvaluationPreset.CHECK_ONLY)

    with pytest.raises(RuntimeConfigurationError, match="does not accept"):
        resolve_evaluation_config(
            EvaluationPreset.CUSTOM,
            preset_inputs=EvaluationPresetInputs(),
            custom_config=custom,
        )


def test_summary_reports_effects_without_private_values() -> None:
    private_path = "/content/drive/MyDrive/private/camera-plan.json"
    custom = CollectionWorkflowConfig(
        persist_to_database=True,
        hud_debug=False,
        view_plan_path=private_path,
        download_outputs=True,
    )
    summary = render_workflow_summary(CollectionPreset.CUSTOM, custom)

    assert "perfil PostgreSQL collection" in summary
    assert "plan privado de vistas configurado" in summary
    assert "descarga explícita" in summary
    assert private_path not in summary


def test_resolver_returns_the_explicit_custom_instance() -> None:
    custom = TrainingWorkflowConfig(
        "seed_bootstrap",
        False,
        True,
        False,
        "reuse_or_create",
        None,
        "reuse_or_create",
        None,
        run_grouped_cross_validation=True,
    )

    assert (
        resolve_training_config(TrainingPreset.CUSTOM, custom_config=custom)
        is custom
    )
