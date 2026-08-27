"""Tests for seed bootstrap and recurrent HITL lifecycle policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaaet_ml.settings import FEATURE_COLS
from vaaet_ml.training.lifecycle import (
    LEGACY_NEUTRAL_FEATURES,
    ModelInputPolicy,
    TrainingMode,
    apply_model_input_policy,
    build_supervision_weights,
    build_training_lifecycle,
    cap_synthetic_congested_weight,
    proxy_memory_weight,
)


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame([{column: 1.0 for column in FEATURE_COLS}] * 4)
    frame["traffic_state"] = [0, 1, 2, 2]
    frame["is_human_validated"] = [False, True, False, False]
    frame["data_origin"] = ["real", "real", "real", "synthetic"]
    return frame


def test_legacy_input_policy_neutralizes_only_quality_features() -> None:
    frame = _frame()
    result = apply_model_input_policy(frame, ModelInputPolicy.LEGACY_V1_BOOTSTRAP)
    assert list(result) == FEATURE_COLS
    assert result.loc[:, LEGACY_NEUTRAL_FEATURES].eq(0).all().all()
    assert result["avg_speed"].eq(1).all()


def test_canonical_input_policy_rejects_unknown_values() -> None:
    frame = _frame()
    frame.loc[0, "speed_measurement_quality"] = np.nan
    with pytest.raises(ValueError, match="unknown feature values"):
        apply_model_input_policy(frame, ModelInputPolicy.CANONICAL_V2)
    legacy = apply_model_input_policy(frame, ModelInputPolicy.LEGACY_V1_BOOTSTRAP)
    assert legacy.loc[0, "speed_measurement_quality"] == 0


@pytest.mark.parametrize(
    ("state", "support", "expected"),
    [(0, 0, 0.5), (0, 150, 0.25), (1, 300, 0.0), (2, 50, 0.25), (2, 100, 0.0)],
)
def test_proxy_memory_decreases_by_human_class_support(
    state: int, support: int, expected: float
) -> None:
    assert proxy_memory_weight(state, {state: support}) == expected


def test_hitl_supervision_keeps_humans_and_decays_proxy() -> None:
    frame = _frame()
    weights, report = build_supervision_weights(frame, TrainingMode.HITL_RETRAINING)
    assert weights[1] == 1.0
    assert 0 < weights[0] < 1
    assert weights[3] < weights[2]
    assert report["human_support"] == {0: 0, 1: 1, 2: 0}


def test_final_synthetic_congested_cap_uses_normal_effective_weight() -> None:
    frame = _frame()
    weights = cap_synthetic_congested_weight(frame, np.array([2.0, 1.0, 1.0, 10.0]))
    assert weights[3] == pytest.approx(1.0)


def test_seed_lifecycle_is_always_pilot() -> None:
    lifecycle = build_training_lifecycle(
        TrainingMode.SEED_BOOTSTRAP,
        ModelInputPolicy.LEGACY_V1_BOOTSTRAP,
        production_eligible=True,
    )
    assert lifecycle == {
        "training_mode": "seed-bootstrap",
        "supervision": "weak-proxy",
        "deployment_stage": "pilot",
        "input_policy": "legacy-v1-bootstrap",
        "production_eligible": False,
    }


def test_seed_lifecycle_rejects_canonical_input_policy() -> None:
    with pytest.raises(ValueError, match="requires the legacy"):
        build_training_lifecycle(
            TrainingMode.SEED_BOOTSTRAP,
            ModelInputPolicy.CANONICAL_V2,
            production_eligible=False,
        )


def test_eligible_hitl_lifecycle_is_production() -> None:
    lifecycle = build_training_lifecycle(
        TrainingMode.HITL_RETRAINING,
        ModelInputPolicy.CANONICAL_V2,
        production_eligible=True,
    )
    assert lifecycle["deployment_stage"] == "production"
    assert lifecycle["production_eligible"] is True
