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
    from vaaet_ml.training.eligibility import CandidateEligibility, evaluate_candidate_eligibility
    from vaaet_ml.training.execution import TrainingFitConfig, build_training_callbacks
    from vaaet_ml.training.lifecycle import (
        HUMAN_SUPPORT_TARGETS,
        LEGACY_NEUTRAL_FEATURES,
        ModelInputPolicy,
        TrainingMode,
        apply_model_input_policy,
        build_supervision_weights,
        build_training_lifecycle,
        cap_synthetic_congested_weight,
        proxy_memory_weight,
    )
    from vaaet_ml.training.observability import (
        TrainingRunReport,
        build_training_run_report,
        compare_training_run_reports,
        list_training_run_reports,
        load_training_run_report,
        write_training_run_report,
    )

__all__ = [
    "BalanceCandidate",
    "BalanceStrategy",
    "CandidateEligibility",
    "HUMAN_SUPPORT_TARGETS",
    "LEGACY_NEUTRAL_FEATURES",
    "ModelInputPolicy",
    "TrainingMode",
    "TrainingFitConfig",
    "TrainingRunReport",
    "apply_model_input_policy",
    "build_balance_candidates",
    "build_training_callbacks",
    "build_supervision_weights",
    "build_training_lifecycle",
    "build_training_run_report",
    "cap_synthetic_congested_weight",
    "compare_training_run_reports",
    "compute_capped_balanced_weights",
    "evaluate_candidate_eligibility",
    "list_training_run_reports",
    "load_training_run_report",
    "proxy_memory_weight",
    "write_training_run_report",
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
        "TrainingFitConfig",
        "build_training_callbacks",
    }:
        from vaaet_ml.training.execution import TrainingFitConfig, build_training_callbacks

        return {
            "TrainingFitConfig": TrainingFitConfig,
            "build_training_callbacks": build_training_callbacks,
        }[name]
    if name in {"CandidateEligibility", "evaluate_candidate_eligibility"}:
        from vaaet_ml.training.eligibility import (
            CandidateEligibility,
            evaluate_candidate_eligibility,
        )

        return {
            "CandidateEligibility": CandidateEligibility,
            "evaluate_candidate_eligibility": evaluate_candidate_eligibility,
        }[name]
    if name in {
        "TrainingRunReport",
        "build_training_run_report",
        "compare_training_run_reports",
        "list_training_run_reports",
        "load_training_run_report",
        "write_training_run_report",
    }:
        from vaaet_ml.training.observability import (
            TrainingRunReport,
            build_training_run_report,
            compare_training_run_reports,
            list_training_run_reports,
            load_training_run_report,
            write_training_run_report,
        )

        return {
            "TrainingRunReport": TrainingRunReport,
            "build_training_run_report": build_training_run_report,
            "compare_training_run_reports": compare_training_run_reports,
            "list_training_run_reports": list_training_run_reports,
            "load_training_run_report": load_training_run_report,
            "write_training_run_report": write_training_run_report,
        }[name]
    if name in {
        "HUMAN_SUPPORT_TARGETS",
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
            HUMAN_SUPPORT_TARGETS,
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
            "HUMAN_SUPPORT_TARGETS": HUMAN_SUPPORT_TARGETS,
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
