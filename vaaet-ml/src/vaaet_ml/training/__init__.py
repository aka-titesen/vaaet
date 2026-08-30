# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Punto de entrada liviano para utilidades de entrenamiento del laboratorio.

Los reexports son diferidos para que contratos como ``holdout`` no inicialicen
scikit-learn ni dependencias de entrenamiento cuando sólo necesitan sus tipos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaaet_ml.training.balancing import (
        BalanceCandidate,
        BalanceStrategy,
        build_balance_candidates,
        compute_capped_balanced_weights,
    )
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

__all__ = [
    "BalanceCandidate",
    "BalanceStrategy",
    "LEGACY_NEUTRAL_FEATURES",
    "ModelInputPolicy",
    "TrainingMode",
    "apply_model_input_policy",
    "build_balance_candidates",
    "build_supervision_weights",
    "build_training_lifecycle",
    "cap_synthetic_congested_weight",
    "compute_capped_balanced_weights",
    "proxy_memory_weight",
]


def __getattr__(name: str) -> object:
    """Resuelve sólo el grupo de utilidades solicitado por un consumidor 4.x."""

    if name in {
        "BalanceCandidate",
        "BalanceStrategy",
        "build_balance_candidates",
        "compute_capped_balanced_weights",
    }:
        from vaaet_ml.training.balancing import (
            BalanceCandidate,
            BalanceStrategy,
            build_balance_candidates,
            compute_capped_balanced_weights,
        )

        return {
            "BalanceCandidate": BalanceCandidate,
            "BalanceStrategy": BalanceStrategy,
            "build_balance_candidates": build_balance_candidates,
            "compute_capped_balanced_weights": compute_capped_balanced_weights,
        }[name]
    if name in {
        "LEGACY_NEUTRAL_FEATURES",
        "ModelInputPolicy",
        "TrainingMode",
        "apply_model_input_policy",
        "build_supervision_weights",
        "build_training_lifecycle",
        "cap_synthetic_congested_weight",
        "proxy_memory_weight",
    }:
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

        return {
            "LEGACY_NEUTRAL_FEATURES": LEGACY_NEUTRAL_FEATURES,
            "ModelInputPolicy": ModelInputPolicy,
            "TrainingMode": TrainingMode,
            "apply_model_input_policy": apply_model_input_policy,
            "build_supervision_weights": build_supervision_weights,
            "build_training_lifecycle": build_training_lifecycle,
            "cap_synthetic_congested_weight": cap_synthetic_congested_weight,
            "proxy_memory_weight": proxy_memory_weight,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
