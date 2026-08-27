# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contrato de ciclo de vida que comparten entrenamiento y serving."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from vaaet.settings import FEATURE_COLS


class TrainingMode(str, Enum):
    """Modo de supervisión declarado por un bundle, no un servicio de entrenamiento."""

    SEED_BOOTSTRAP = "seed-bootstrap"
    HITL_RETRAINING = "hitl-retraining"


class ModelInputPolicy(str, Enum):
    """Política de disponibilidad de features aplicada igual en train y serve."""

    LEGACY_V1_BOOTSTRAP = "legacy-v1-bootstrap"
    CANONICAL_V2 = "canonical-v2"


LEGACY_NEUTRAL_FEATURES: tuple[str, ...] = (
    "speed_measurement_quality",
    "near_zero_motion_ratio",
    "stationary_confirmed_ratio",
)


def apply_model_input_policy(
    frame: pd.DataFrame,
    policy: ModelInputPolicy | str,
) -> pd.DataFrame:
    """Return the canonical model matrix with explicit legacy neutralization."""

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


def build_training_lifecycle(
    mode: TrainingMode | str,
    input_policy: ModelInputPolicy | str,
    *,
    production_eligible: bool,
) -> dict[str, object]:
    """Build the bundle lifecycle block without coupling it to training I/O."""

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
