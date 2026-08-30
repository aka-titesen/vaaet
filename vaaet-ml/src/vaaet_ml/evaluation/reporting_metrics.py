# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Métricas y decisiones de evaluación sin gráficos ni dependencias de notebook."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import sqrt

import numpy as np
import pandas as pd
from vaaet.settings import STATE_LABELS

CONFUSION_COST = np.array(
    [[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]]
)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return (max(0.0, centre - radius / denominator), min(1.0, centre + radius / denominator))


def build_classification_support_table(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> pd.DataFrame:
    """Reporta soporte y Wilson 95% por cada estado aprendible."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    rows = [
        _support_row(code, truth, predicted)
        for code in (0, 1, 2)
    ]
    return pd.DataFrame(rows)


def _support_row(code: int, truth: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    true_positive = int(((truth == code) & (predicted == code)).sum())
    actual = int((truth == code).sum())
    predicted_count = int((predicted == code).sum())
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / actual if actual else 0.0
    return {
        "traffic_state": code,
        "state_label": STATE_LABELS[code],
        "support": actual,
        "predicted": predicted_count,
        "precision": precision,
        "precision_ci_95": _wilson_interval(true_positive, predicted_count),
        "recall": recall,
        "recall_ci_95": _wilson_interval(true_positive, actual),
    }


def expected_confusion_cost(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Calcula el costo operacional asimétrico medio de los tres estados estables."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional and equally sized.")
    if len(truth) == 0:
        return 0.0
    if not set(truth).union(predicted).issubset({0, 1, 2}):
        raise ValueError("Confusion cost is defined only for Normal/Reduced/Congested.")
    return float(CONFUSION_COST[truth, predicted].mean())


def expected_calibration_error(
    y_true: Sequence[int], probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Calcula ECE top-label multiclase sin asumir que softmax está calibrado."""

    truth = np.asarray(y_true, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.shape != (len(truth), 3):
        raise ValueError("probabilities must have shape (records, 3).")
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == truth
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def select_validation_decision_policy(
    frame: pd.DataFrame,
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    temperature: float,
) -> dict[str, object]:
    """Selecciona umbrales sólo por costo de validación, nunca por métricas test."""

    from vaaet.inference.traffic_state import apply_stable_state_policy

    truth = np.asarray(y_true, dtype=int)
    candidates = (0.55, 0.60, 0.65, 0.70, 0.75)
    best_cost = float("inf")
    best: tuple[tuple[float, float, float], float] | None = None
    for thresholds, margin in product(product(candidates, repeat=3), (0.05, 0.10, 0.15)):
        result = apply_stable_state_policy(
            frame,
            probabilities,
            class_thresholds=dict(enumerate(thresholds)),
            minimum_margin=margin,
        )
        cost = expected_confusion_cost(truth, result["traffic_state"].to_numpy())
        if cost < best_cost:
            best_cost = cost
            best = (thresholds, margin)
    if best is None:
        raise ValueError("Validation set is empty; decision policy cannot be selected.")
    thresholds, margin = best
    return {
        "class_thresholds": {str(code): value for code, value in enumerate(thresholds)},
        "minimum_probability_margin": margin,
        "temperature": float(temperature),
        "validation_expected_confusion_cost": best_cost,
    }


__all__ = [
    "build_classification_support_table",
    "expected_calibration_error",
    "expected_confusion_cost",
    "select_validation_decision_policy",
]
