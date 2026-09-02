# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contratos compartidos por el bootstrap semilla y el entrenamiento HITL."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from vaaet.lifecycle import (
    LEGACY_NEUTRAL_FEATURES,
    ModelInputPolicy,
    TrainingMode,
    apply_model_input_policy,
    build_training_lifecycle,
)

__all__ = [
    "HUMAN_SUPPORT_TARGETS",
    "LEGACY_NEUTRAL_FEATURES",
    "ModelInputPolicy",
    "TrainingMode",
    "apply_model_input_policy",
    "build_training_lifecycle",
    "build_supervision_weights",
    "cap_synthetic_congested_weight",
    "proxy_memory_weight",
]

HUMAN_SUPPORT_TARGETS: Mapping[int, int] = MappingProxyType({0: 300, 1: 300, 2: 100})


def proxy_memory_weight(state: int, human_support: Mapping[int, int]) -> float:
    """Calcula el peso determinista de memoria proxy por clase para HITL."""
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
    """Calcula pesos human-first antes de aplicar el balance de clases."""
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
        "human_support_targets": dict(HUMAN_SUPPORT_TARGETS),
        "human_support_progress": {
            state: min(1.0, support[state] / HUMAN_SUPPORT_TARGETS[state])
            for state in HUMAN_SUPPORT_TARGETS
        },
        "human_support_deficit": {
            state: max(HUMAN_SUPPORT_TARGETS[state] - support[state], 0)
            for state in HUMAN_SUPPORT_TARGETS
        },
        "proxy_memory_weight": {
            state: 1.0
            if active_mode is TrainingMode.SEED_BOOTSTRAP
            else proxy_memory_weight(state, support)
            for state in HUMAN_SUPPORT_TARGETS
        },
        "effective_weight_by_class": _effective_weight_by_class(
            labels=labels,
            human=human,
            synthetic=synthetic,
            weights=weights,
        ),
        "synthetic_multiplier": float(synthetic_multiplier),
        "synthetic_congested_effective_weight": float(
            weights[congested_synthetic.to_numpy()].sum()
        ),
        "synthetic_congested_limit": synthetic_limit,
    }
    return weights, report


def _effective_weight_by_class(
    *,
    labels: pd.Series,
    human: pd.Series,
    synthetic: pd.Series,
    weights: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Separa el aporte efectivo sin usar el holdout para alterar la memoria."""

    result: dict[int, dict[str, float]] = {}
    for state in HUMAN_SUPPORT_TARGETS:
        state_mask = labels.eq(state)
        source_masks = {
            "human": state_mask & human,
            "synthetic": state_mask & ~human & synthetic,
            "proxy": state_mask & ~human & ~synthetic,
        }
        result[state] = {
            source: float(weights[mask.to_numpy()].sum()) for source, mask in source_masks.items()
        }
    return result


def cap_synthetic_congested_weight(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    max_fraction_of_normal: float = 0.5,
) -> np.ndarray:
    """Acota el peso sintético efectivo de ``Congested`` después del balance."""
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
