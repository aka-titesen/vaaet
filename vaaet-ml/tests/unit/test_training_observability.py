# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contratos de los informes agregados e inmutables de entrenamiento."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vaaet.lifecycle import ModelInputPolicy, build_training_lifecycle

import vaaet_ml.training.observability as observability
from vaaet_ml.data.training_input_lock import create_training_input_lock
from vaaet_ml.evaluation.reporting import build_classification_support_table
from vaaet_ml.evaluation.training_observability_visuals import save_training_run_diagnostics
from vaaet_ml.exceptions import TrainingStabilityError
from vaaet_ml.training.eligibility import CandidateEligibility
from vaaet_ml.training.execution import TrainingFitConfig
from vaaet_ml.training.holdout_contract import HUMAN_HOLDOUT_CONTRACT
from vaaet_ml.training.lifecycle import TrainingMode
from vaaet_ml.training.observability import (
    TRAINING_OBSERVABILITY_REPORT_FILE,
    TRAINING_OBSERVABILITY_SUMMARY_FILE,
    build_training_evaluation_evidence,
    build_training_run_report,
    compare_training_run_reports,
    list_training_run_reports,
    load_training_run_report,
    write_training_run_report,
)


def _build_report(
    output_root: Path,
    *,
    run_id: str | None = None,
    holdout_fingerprint: str = "b" * 64,
    f1_macro: float = 1.0,
    runtime: dict[str, object] | None = None,
    decision_policy: dict[str, object] | None = None,
):
    active_run_id = run_id or str(uuid.uuid4())
    lock = create_training_input_lock(
        output_root,
        training_pipeline_run_id=active_run_id,
        training_mode=TrainingMode.HITL_RETRAINING.value,
        seed_snapshot={"fingerprint": "a" * 64},
        hitl_catalog={
            "revision": 2,
            "package_ids": [str(uuid.uuid5(uuid.NAMESPACE_URL, active_run_id))],
        },
        human_holdout={
            "contract": HUMAN_HOLDOUT_CONTRACT,
            "snapshot_id": str(uuid.uuid5(uuid.NAMESPACE_URL, holdout_fingerprint)),
            "generation": 1,
            "fingerprint": holdout_fingerprint,
        },
        result_rows={"train": 700, "validation": 150, "test": 150},
        resolution={"duplicates": 0, "corrections": 0},
    )
    truth = np.asarray([0, 1, 2], dtype=int)
    probabilities = np.asarray(
        [[0.98, 0.01, 0.01], [0.02, 0.96, 0.02], [0.01, 0.02, 0.97]],
        dtype=float,
    )
    supervision = {
        "training_mode": TrainingMode.HITL_RETRAINING.value,
        "human_support": {0: 300, 1: 300, 2: 100},
        "human_support_targets": {0: 300, 1: 300, 2: 100},
        "human_support_progress": {0: 1.0, 1: 1.0, 2: 1.0},
        "human_support_deficit": {0: 0, 1: 0, 2: 0},
        "proxy_memory_weight": {0: 0.0, 1: 0.0, 2: 0.0},
        "effective_weight_by_class": {
            0: {"human": 300.0, "proxy": 0.0, "synthetic": 0.0},
            1: {"human": 300.0, "proxy": 0.0, "synthetic": 0.0},
            2: {"human": 100.0, "proxy": 0.0, "synthetic": 0.0},
        },
        "synthetic_multiplier": 0.35,
        "synthetic_congested_effective_weight": 0.0,
        "synthetic_congested_limit": 150.0,
    }
    return build_training_run_report(
        training_input_lock=lock,
        training_lifecycle=build_training_lifecycle(
            TrainingMode.HITL_RETRAINING,
            ModelInputPolicy.CANONICAL_V2,
            production_eligible=True,
        ),
        fit_config=TrainingFitConfig(random_seed=42),
        supervision_report=supervision,
        partition_rows={"train": 700, "validation": 150, "test": 150},
        selected_balance_strategy="class-weights",
        balance_candidates=[{"strategy": "class-weights", "validation_score": 0.95}],
        training_history=SimpleNamespace(
            history={
                "loss": [0.6, 0.4],
                "val_loss": [0.7, 0.45],
                "accuracy": [0.7, 0.85],
                "val_accuracy": [0.65, 0.82],
            }
        ),
        direct_metrics={"f1_macro": 1.0, "direct_normal_congested_error": 0.0},
        policy_metrics={
            "f1_macro": f1_macro,
            "expected_confusion_cost": 0.0,
            "ece": 0.02,
            "brier_score": 0.01,
        },
        incident_metrics={"negative_exposure_hours": 300.0, "false_candidates_per_hour": 0.0},
        support_table=build_classification_support_table(truth, truth),
        evaluation_evidence=build_training_evaluation_evidence(
            truth, truth, truth, probabilities
        ),
        eligibility=CandidateEligibility(
            metric_gates={"f1_macro": True, "ece": True},
            promotion_blockers=(),
            human_holdout=True,
            congested_minutes=100,
            congested_clips=20,
            production_eligible=True,
        ),
        decision_policy=decision_policy
        or {"minimum_probability_margin": 0.1, "temperature": 1.0},
        runtime=runtime
        or {
            "git_commit": "1234abc",
            "python_version": "3.12.0",
            "tensorflow_version": "2.20.0",
            "keras_version": "3.12.0",
            "framework_gpu_available": True,
            "total_ram_gib": 12.0,
            "available_ram_gib": 8.0,
            "content_free_gib": 40.0,
            "declared_extras": ("training", "visualization"),
        },
        model_version="mlp-v4.5.3",
    )


def test_report_is_immutable_and_contains_only_aggregated_evidence(tmp_path: Path) -> None:
    report = _build_report(tmp_path)

    persisted = write_training_run_report(tmp_path, report)
    report_path = tmp_path / persisted.run_id / TRAINING_OBSERVABILITY_REPORT_FILE
    summary_path = tmp_path / persisted.run_id / TRAINING_OBSERVABILITY_SUMMARY_FILE

    assert persisted.path == report_path.resolve()
    assert load_training_run_report(report_path).document == persisted.document
    assert list_training_run_reports(tmp_path) == (persisted,)
    summary = summary_path.read_text(encoding="utf-8")
    assert "la decisión sigue siendo humana" in summary
    assert str(tmp_path) not in summary
    assert write_training_run_report(tmp_path, report) == persisted

    altered = _build_report(tmp_path, run_id=persisted.run_id, f1_macro=0.9)
    with pytest.raises(TrainingStabilityError, match="different observability report"):
        write_training_run_report(tmp_path, altered)


def test_report_rejects_tampering_and_sensitive_runtime_values(tmp_path: Path) -> None:
    persisted = write_training_run_report(tmp_path, _build_report(tmp_path))
    assert persisted.path is not None
    raw = json.loads(persisted.path.read_text(encoding="utf-8"))
    raw["metrics"]["policy"]["f1_macro"] = 0.1
    persisted.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(TrainingStabilityError, match="fingerprint does not match"):
        load_training_run_report(persisted.path)
    with pytest.raises(ValueError, match="unsupported fields"):
        _build_report(tmp_path, runtime={"private_path": "C:/private"})
    with pytest.raises(ValueError, match="sensitive field name"):
        _build_report(tmp_path, decision_policy={"api_token": "not-persisted"})


def test_comparison_requires_a_compatible_frozen_holdout(tmp_path: Path) -> None:
    reference = _build_report(tmp_path, f1_macro=0.9)
    current = _build_report(tmp_path, f1_macro=1.0)

    comparison = compare_training_run_reports(current, reference)

    assert comparison.comparable is True
    assert comparison.metric_deltas["f1_macro"] == pytest.approx(0.1)
    incompatible = compare_training_run_reports(
        current,
        _build_report(tmp_path, holdout_fingerprint="c" * 64),
    )
    assert incompatible.comparable is False
    assert incompatible.metric_deltas == {}
    assert "frozen human holdout fingerprints differ" in incompatible.reasons


def test_diagnostics_are_idempotent_and_use_only_report_aggregates(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    persisted = write_training_run_report(tmp_path, _build_report(tmp_path))

    first = save_training_run_diagnostics(persisted, tmp_path)
    second = save_training_run_diagnostics(persisted, tmp_path)

    assert first == second
    assert set(first) == {"optimization-curves", "test-quality", "supervision-governance"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in first.values())


@pytest.mark.parametrize(
    ("actual", "direct", "policy", "probabilities", "message"),
    [
        ([0], [0, 1], [0], [[1.0, 0.0, 0.0]], "aligned"),
        ([3], [3], [0], [[1.0, 0.0, 0.0]], "three stable"),
        ([0], [0], [3], [[1.0, 0.0, 0.0]], "invalid policy"),
        ([0], [0], [0], [[0.8, 0.0, 0.0]], "finite normalized"),
    ],
)
def test_evaluation_evidence_fails_closed_for_malformed_predictions(
    actual: list[int],
    direct: list[int],
    policy: list[int],
    probabilities: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_training_evaluation_evidence(actual, direct, policy, np.asarray(probabilities))


def test_report_helpers_reject_invalid_runtime_and_unsafe_aggregates(tmp_path: Path) -> None:
    assert observability._numeric_metrics({"missing": None, "records": 2}) == {
        "missing": None,
        "records": 2,
    }
    with pytest.raises(ValueError, match="not boolean"):
        observability._numeric_metrics({"f1_macro": True})
    with pytest.raises(ValueError, match="finite and non-negative"):
        observability._numeric_metrics({"f1_macro": -0.1})
    with pytest.raises(ValueError, match="sequence"):
        observability._runtime_evidence({"declared_extras": "training"})
    with pytest.raises(ValueError, match="must be boolean"):
        observability._runtime_evidence({"framework_gpu_available": "yes"})
    with pytest.raises(ValueError, match="non-negative number"):
        observability._runtime_evidence({"total_ram_gib": -1})
    with pytest.raises(ValueError, match="sensitive path"):
        observability._safe_aggregate_mapping({"summary": ["C:/private"]}, label="test")
    with pytest.raises(ValueError, match="hexadecimal revision"):
        observability._safe_identifier("not-a-sha", "git_commit")

    report = _build_report(tmp_path)
    assert observability._holdout_descriptor(None) is None
    assert observability._optional_safe_aggregate_mapping(None, label="cross_validation") is None
    with pytest.raises(ValueError, match="between 1 and 100"):
        list_training_run_reports(tmp_path, limit=0)
    with pytest.raises(TrainingStabilityError, match="could not be read"):
        load_training_run_report(tmp_path / "missing.json")
    (tmp_path / "array.json").write_text("[]", encoding="utf-8")
    with pytest.raises(TrainingStabilityError, match="JSON object"):
        load_training_run_report(tmp_path / "array.json")
    assert report.document["contract"] == "vaaet-training-observability-report-v1"
