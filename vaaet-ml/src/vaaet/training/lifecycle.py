"""Contracts shared by the one-time seed bootstrap and recurrent HITL training."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from vaaet.settings import FEATURE_COLS


class TrainingMode(str, Enum):
    """Explicit supervised-data lifecycle selected by the training notebook."""

    SEED_BOOTSTRAP = "seed-bootstrap"
    HITL_RETRAINING = "hitl-retraining"


class ModelInputPolicy(str, Enum):
    """Feature-availability policy stored in and enforced by the model bundle."""

    LEGACY_V1_BOOTSTRAP = "legacy-v1-bootstrap"
    CANONICAL_V2 = "canonical-v2"


LEGACY_NEUTRAL_FEATURES: tuple[str, ...] = (
    "speed_measurement_quality",
    "near_zero_motion_ratio",
    "stationary_confirmed_ratio",
)
HUMAN_SUPPORT_TARGETS: Mapping[int, int] = MappingProxyType({0: 300, 1: 300, 2: 100})


def apply_model_input_policy(
    frame: pd.DataFrame,
    policy: ModelInputPolicy | str,
) -> pd.DataFrame:
    """Return the canonical model matrix with train/serve-identical neutralization."""
    active_policy = ModelInputPolicy(policy)
    missing = [column for column in FEATURE_COLS if column not in frame]
    if missing:
        raise ValueError(f"Model input is missing canonical features: {missing}")
    matrix = frame.loc[:, FEATURE_COLS].copy()
    if active_policy is ModelInputPolicy.LEGACY_V1_BOOTSTRAP:
        matrix.loc[:, LEGACY_NEUTRAL_FEATURES] = 0.0
    if matrix.isna().any().any():
        columns = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"Model input contains unknown feature values: {columns}")
    return matrix


def proxy_memory_weight(state: int, human_support: Mapping[int, int]) -> float:
    """Return the deterministic class-specific proxy replay weight for HITL."""
    if state not in HUMAN_SUPPORT_TARGETS:
        raise ValueError("Proxy memory is defined only for stable states 0, 1, and 2.")
    support = max(int(human_support.get(state, 0)), 0)
    target = HUMAN_SUPPORT_TARGETS[state]
    return 0.5 * max(0.0, 1.0 - support / target)


def build_supervision_weights(
    frame: pd.DataFrame,
    mode: TrainingMode | str,
    *,
    synthetic_multiplier: float = 0.35,
    max_synthetic_congested_fraction_of_normal: float = 0.5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build human-first proxy replay weights before class balancing is applied."""
    active_mode = TrainingMode(mode)
    if "traffic_state" not in frame:
        raise ValueError("Supervised frame must contain traffic_state.")
    labels = pd.to_numeric(frame["traffic_state"], errors="raise").astype(int)
    if not labels.isin((0, 1, 2)).all():
        raise ValueError("Stable MLP supervision may contain only states 0, 1, and 2.")
    human = frame.get(
        "is_human_validated", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    support = {
        state: int((human & labels.eq(state)).sum()) for state in HUMAN_SUPPORT_TARGETS
    }
    weights = np.ones(len(frame), dtype=float)
    if active_mode is TrainingMode.HITL_RETRAINING:
        for state in HUMAN_SUPPORT_TARGETS:
            weights[(~human & labels.eq(state)).to_numpy()] = proxy_memory_weight(
                state, support
            )

    synthetic = frame.get(
        "data_origin", pd.Series("real", index=frame.index)
    ).eq("synthetic")
    weights[synthetic.to_numpy()] *= float(synthetic_multiplier)

    congested_synthetic = synthetic & labels.eq(2)
    synthetic_effective = float(weights[congested_synthetic.to_numpy()].sum())
    normal_effective = float(weights[labels.eq(0).to_numpy()].sum())
    synthetic_limit = normal_effective * float(max_synthetic_congested_fraction_of_normal)
    if synthetic_effective > synthetic_limit >= 0:
        if synthetic_limit == 0:
            weights[congested_synthetic.to_numpy()] = 0.0
        else:
            weights[congested_synthetic.to_numpy()] *= synthetic_limit / synthetic_effective

    report: dict[str, object] = {
        "training_mode": active_mode.value,
        "human_support": support,
        "proxy_memory_weight": {
            state: 1.0
            if active_mode is TrainingMode.SEED_BOOTSTRAP
            else proxy_memory_weight(state, support)
            for state in HUMAN_SUPPORT_TARGETS
        },
        "synthetic_multiplier": float(synthetic_multiplier),
        "synthetic_congested_effective_weight": float(
            weights[congested_synthetic.to_numpy()].sum()
        ),
        "synthetic_congested_limit": synthetic_limit,
    }
    return weights, report


def cap_synthetic_congested_weight(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    max_fraction_of_normal: float = 0.5,
) -> np.ndarray:
    """Cap final effective synthetic Congested weight after class balancing."""
    result = np.asarray(weights, dtype=float).copy()
    if result.shape != (len(frame),):
        raise ValueError("Supervision weights must match the training frame length.")
    labels = pd.to_numeric(frame["traffic_state"], errors="raise").astype(int)
    synthetic = frame.get(
        "data_origin", pd.Series("real", index=frame.index)
    ).eq("synthetic")
    mask = (synthetic & labels.eq(2)).to_numpy()
    current = float(result[mask].sum())
    limit = float(result[labels.eq(0).to_numpy()].sum()) * float(max_fraction_of_normal)
    if current > limit:
        result[mask] = 0.0 if limit <= 0 else result[mask] * limit / current
    return result


def build_training_lifecycle(
    mode: TrainingMode | str,
    input_policy: ModelInputPolicy | str,
    *,
    production_eligible: bool,
) -> dict[str, object]:
    """Build the manifest lifecycle block used by training and inference."""
    active_mode = TrainingMode(mode)
    active_policy = ModelInputPolicy(input_policy)
    if active_mode is TrainingMode.SEED_BOOTSTRAP:
        if active_policy is not ModelInputPolicy.LEGACY_V1_BOOTSTRAP:
            raise ValueError("Seed bootstrap requires the legacy-v1-bootstrap input policy.")
        deployment_stage = "pilot"
        supervision = "weak-proxy"
        production_eligible = False
    else:
        deployment_stage = "production" if production_eligible else "candidate"
        supervision = "human-validated-with-proxy-memory"
    return {
        "training_mode": active_mode.value,
        "supervision": supervision,
        "deployment_stage": deployment_stage,
        "input_policy": active_policy.value,
        "production_eligible": bool(production_eligible),
    }
