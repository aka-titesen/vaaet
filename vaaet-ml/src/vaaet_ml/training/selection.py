# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Validation-only selection of conservative balance candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from vaaet.calibration import apply_temperature_scaling, fit_temperature
from vaaet.inference.traffic_state import classify_telemetry_dataframe

from vaaet_ml.evaluation.reporting import expected_confusion_cost, select_validation_decision_policy
from vaaet_ml.training.balancing import (
    BalanceCandidate,
    BalanceStrategy,
    compute_capped_balanced_weights,
)
from vaaet_ml.training.lifecycle import ModelInputPolicy, cap_synthetic_congested_weight


@dataclass(frozen=True)
class BalanceSelection:
    """The selected model and its validation-only comparison evidence."""

    strategy: BalanceStrategy
    model: object
    history: object
    sample_weight: np.ndarray
    class_weights: dict[int, float]
    report: pd.DataFrame


def select_balance_candidate(
    *,
    candidates: Mapping[BalanceStrategy, BalanceCandidate],
    train_frame: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    validation_frame: pd.DataFrame,
    scaler: object,
    input_policy: ModelInputPolicy,
    input_features: int,
    output_classes: int,
    random_seed: int,
    callbacks: Sequence[object],
    clear_session: Callable[[], None],
    set_random_seed: Callable[[int], None],
) -> BalanceSelection:
    """Fit candidate MLPs and choose the lowest-risk validation result."""

    from vaaet_ml.training.modeling import build_traffic_state_mlp

    records: list[dict[str, float | int | str]] = []
    trained: dict[BalanceStrategy, tuple[object, object, np.ndarray, dict[int, float]]] = {}
    for strategy, candidate in candidates.items():
        clear_session()
        set_random_seed(random_seed)
        positions = candidate.row_positions
        candidate_frame = train_frame.iloc[positions].copy()
        candidate_y = y_train[positions]
        sample_weight, class_weights = compute_capped_balanced_weights(
            candidate_y, candidate.supervision_weights
        )
        if strategy is BalanceStrategy.SYNTHETIC_CONGESTION:
            sample_weight = cap_synthetic_congested_weight(candidate_frame, sample_weight)
        model = build_traffic_state_mlp(
            input_features=input_features, output_classes=output_classes
        )
        history = model.fit(
            x_train[positions],
            candidate_y,
            epochs=120,
            batch_size=32,
            validation_data=(x_validation, y_validation),
            sample_weight=sample_weight,
            callbacks=list(callbacks),
            verbose=0,
        )
        raw_validation_proba = model.predict(x_validation, verbose=0)
        temperature = fit_temperature(raw_validation_proba, y_validation)
        validation_proba = apply_temperature_scaling(raw_validation_proba, temperature)
        policy = select_validation_decision_policy(
            validation_frame, y_validation, validation_proba, temperature=temperature
        )
        classified = classify_telemetry_dataframe(
            validation_frame, model, scaler, decision_policy=policy, input_policy=input_policy
        )
        predicted = classified["traffic_state"].to_numpy(dtype=int)
        cost = expected_confusion_cost(y_validation, predicted)
        false_congested = float(((predicted == 2) & (y_validation != 2)).mean())
        f1_macro = _macro_f1(y_validation, predicted)
        records.append(
            {
                "strategy": strategy.value,
                "training_rows": len(positions),
                "validation_f1_macro": f1_macro,
                "validation_confusion_cost": cost,
                "validation_false_congested_rate": false_congested,
                "selection_score": cost + 4.0 * false_congested,
            }
        )
        trained[strategy] = (model, history, sample_weight, class_weights)
    report = pd.DataFrame(records).sort_values(
        ["selection_score", "validation_false_congested_rate", "validation_f1_macro"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    strategy = BalanceStrategy(report.iloc[0]["strategy"])
    model, history, sample_weight, class_weights = trained[strategy]
    return BalanceSelection(strategy, model, history, sample_weight, class_weights, report)


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
