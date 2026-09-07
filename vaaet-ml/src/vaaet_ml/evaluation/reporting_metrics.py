# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Métricas y decisiones de evaluación sin gráficos ni dependencias de notebook."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_score, recall_score
from vaaet.calibration import multiclass_brier_score
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
    y_true: Sequence[int], y_pred: Sequence[int], *, clip_ids: Sequence[object] | None = None
) -> pd.DataFrame:
    """Reporta soporte por filas y clips para cada estado aprendible."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    groups = np.asarray(clip_ids, dtype=object) if clip_ids is not None else None
    if groups is not None and groups.shape != truth.shape:
        raise ValueError("clip_ids must align with y_true.")
    rows = [
        _support_row(code, truth, predicted, groups)
        for code in (0, 1, 2)
    ]
    return pd.DataFrame(rows)


def _support_row(
    code: int,
    truth: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray | None,
) -> dict[str, object]:
    true_positive = int(((truth == code) & (predicted == code)).sum())
    actual = int((truth == code).sum())
    predicted_count = int((predicted == code).sum())
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / actual if actual else 0.0
    return {
        "traffic_state": code,
        "state_label": STATE_LABELS[code],
        "support": actual,
        "support_clips": (
            int(len(set(groups[truth == code].tolist()))) if groups is not None else pd.NA
        ),
        "predicted": predicted_count,
        "precision": precision,
        "precision_ci_95": _wilson_interval(true_positive, predicted_count),
        "recall": recall,
        "recall_ci_95": _wilson_interval(true_positive, actual),
    }


def grouped_classification_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    clip_ids: Sequence[object],
    *,
    probabilities: np.ndarray | None = None,
    samples: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estima intervalos del 95 % remuestreando clips completos.

    Una métrica por clase sólo es suficiente si al menos el 95 % de las
    réplicas contiene el soporte necesario para calcularla.
    """

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    groups = np.asarray(clip_ids, dtype=object)
    if truth.ndim != 1 or predicted.shape != truth.shape or groups.shape != truth.shape:
        raise ValueError("Grouped bootstrap inputs must be aligned one-dimensional arrays.")
    if samples < 1 or not len(truth):
        raise ValueError("Grouped bootstrap requires records and at least one sample.")
    probability_values = None if probabilities is None else np.asarray(probabilities, dtype=float)
    if probability_values is not None and probability_values.shape != (len(truth), 3):
        raise ValueError("probabilities must have shape (records, 3).")
    unique_groups = pd.unique(groups)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {name: [] for name in _metric_names(probability_values)}
    for _ in range(samples):
        sampled_groups = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        for name, value in _classification_metrics(
            truth[indices],
            predicted[indices],
            None if probability_values is None else probability_values[indices],
        ).items():
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    point = _classification_metrics(truth, predicted, probability_values)
    rows: list[dict[str, object]] = []
    for name, point_value in point.items():
        metric_values = values[name]
        evaluable_fraction = len(metric_values) / samples
        rows.append(
            {
                "metric": name,
                "value": point_value,
                "ci_95_low": (
                    float(np.quantile(metric_values, 0.025)) if metric_values else np.nan
                ),
                "ci_95_high": (
                    float(np.quantile(metric_values, 0.975)) if metric_values else np.nan
                ),
                "evaluable_fraction": evaluable_fraction,
                "sufficient": evaluable_fraction >= 0.95,
                "direction": (
                    "lower-is-better"
                    if name in {"normal_congested_error", "ece", "brier_score"}
                    else "higher-is-better"
                ),
            }
        )
    return pd.DataFrame(rows)


def false_alert_rate_upper_bound(alerts: int, negative_exposure_hours: float) -> float:
    """Calcula el límite superior unilateral del 95 % para una tasa Poisson."""

    if alerts < 0 or negative_exposure_hours <= 0:
        raise ValueError("False-alert evidence requires non-negative alerts and positive hours.")
    upper_count = 0.5 * float(chi2.ppf(0.95, 2 * (alerts + 1)))
    return upper_count / float(negative_exposure_hours)


def _metric_names(probabilities: np.ndarray | None) -> tuple[str, ...]:
    base = (
        "f1_macro", "precision_normal", "precision_reduced", "precision_congested",
        "recall_normal", "recall_reduced", "recall_congested",
        "normal_congested_error",
    )
    return (*base, "ece", "brier_score") if probabilities is not None else base


def _classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "f1_macro": (
            float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
            if set(truth) == {0, 1, 2}
            else None
        ),
        "normal_congested_error": (
            float(
                (((truth == 0) & (predicted == 2)) | ((truth == 2) & (predicted == 0))).mean()
            )
            if {0, 2}.issubset(set(truth))
            else None
        ),
    }
    for code, label in ((0, "normal"), (1, "reduced"), (2, "congested")):
        result[f"precision_{label}"] = (
            float(precision_score(truth, predicted, labels=[code], average="macro", zero_division=0))
            if (predicted == code).any()
            else None
        )
        result[f"recall_{label}"] = (
            float(recall_score(truth, predicted, labels=[code], average="macro", zero_division=0))
            if (truth == code).any()
            else None
        )
    if probabilities is not None:
        result["ece"] = expected_calibration_error(truth, probabilities)
        result["brier_score"] = multiclass_brier_score(truth, probabilities)
    return result


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
    "false_alert_rate_upper_bound",
    "grouped_classification_intervals",
    "select_validation_decision_policy",
]
