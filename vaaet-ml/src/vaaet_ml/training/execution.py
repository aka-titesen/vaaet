# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Ejecución Keras reproducible y validada para el entrenamiento tabular."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vaaet_ml.exceptions import TrainingStabilityError

__all__ = [
    "TrainingFitConfig",
    "build_training_callbacks",
    "reset_training_state",
    "set_keras_random_seed",
    "validate_training_history",
    "validate_training_inputs",
]


@dataclass(frozen=True)
class TrainingFitConfig:
    """Parámetros estables del ajuste canónico, sin definir la arquitectura."""

    random_seed: int
    epochs: int = 120
    batch_size: int = 32
    early_stopping_patience: int = 15
    reduce_learning_rate_patience: int = 5
    reduce_learning_rate_factor: float = 0.5
    minimum_learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        for name in (
            "epochs",
            "batch_size",
            "early_stopping_patience",
            "reduce_learning_rate_patience",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0 < self.reduce_learning_rate_factor < 1:
            raise ValueError("reduce_learning_rate_factor must be between zero and one.")
        if self.minimum_learning_rate <= 0:
            raise ValueError("minimum_learning_rate must be positive.")


def build_training_callbacks(config: TrainingFitConfig) -> tuple[object, ...]:
    """Crea callbacks nuevos para no compartir estado entre ajustes independientes."""

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
    except ImportError as exc:  # pragma: no cover - depende del extra de entrenamiento.
        raise RuntimeError("Training callbacks require the vaaet-ml training extra.") from exc
    return (
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.reduce_learning_rate_patience,
            factor=config.reduce_learning_rate_factor,
            min_lr=config.minimum_learning_rate,
            verbose=0,
        ),
        TerminateOnNaN(),
    )


def set_keras_random_seed(random_seed: int) -> None:
    """Sincroniza las semillas de Python, NumPy y TensorFlow para una corrida."""

    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depende del extra de entrenamiento.
        raise RuntimeError("Training reproducibility requires the vaaet-ml training extra.") from exc
    keras.utils.set_random_seed(random_seed)


def reset_training_state(
    config: TrainingFitConfig,
    *,
    clear_session: Callable[[], None],
    set_random_seed: Callable[[int], None] = set_keras_random_seed,
) -> None:
    """Limpia el estado global antes de cada candidato o fold independiente."""

    clear_session()
    set_random_seed(config.random_seed)


def validate_training_inputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    input_features: int,
    output_classes: int,
) -> None:
    """Rechaza matrices o targets que podrían invalidar silenciosamente el ajuste."""

    _validate_feature_matrix("train", x_train, input_features)
    _validate_feature_matrix("validation", x_validation, input_features)
    _validate_labels("train", y_train, len(x_train), output_classes)
    _validate_labels("validation", y_validation, len(x_validation), output_classes)


def validate_training_history(history: object) -> dict[str, tuple[float, ...]]:
    """Verifica que la historia contenga series finitas de train y validation."""

    raw_history = getattr(history, "history", None)
    if not isinstance(raw_history, Mapping):
        raise TrainingStabilityError("Keras training did not return a metric history.")
    required = ("loss", "val_loss", "accuracy", "val_accuracy")
    missing = [name for name in required if name not in raw_history]
    if missing:
        raise TrainingStabilityError(
            f"Keras training history is missing required series: {', '.join(missing)}."
        )
    normalized: dict[str, tuple[float, ...]] = {}
    for name in required:
        values = _finite_series(name, raw_history[name])
        normalized[name] = values
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise TrainingStabilityError("Keras training history series have inconsistent lengths.")
    return normalized


def _validate_feature_matrix(name: str, values: np.ndarray, input_features: int) -> None:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise TrainingStabilityError(f"{name} features must be a non-empty two-dimensional matrix.")
    if matrix.shape[1] != input_features:
        raise TrainingStabilityError(
            f"{name} features must contain {input_features} columns; received {matrix.shape[1]}."
        )
    try:
        finite = np.isfinite(matrix.astype(float, copy=False)).all()
    except (TypeError, ValueError) as exc:
        raise TrainingStabilityError(f"{name} features must be numeric.") from exc
    if not finite:
        raise TrainingStabilityError(f"{name} features contain NaN or infinite values.")


def _validate_labels(name: str, labels: np.ndarray, rows: int, output_classes: int) -> None:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != rows or values.size == 0:
        raise TrainingStabilityError(f"{name} labels must be a non-empty vector aligned with features.")
    try:
        numeric = values.astype(int, copy=False)
    except (TypeError, ValueError) as exc:
        raise TrainingStabilityError(f"{name} labels must be integer state codes.") from exc
    if not np.array_equal(values, numeric) or not np.isin(numeric, np.arange(output_classes)).all():
        raise TrainingStabilityError(
            f"{name} labels must contain only state codes from zero to {output_classes - 1}."
        )


def _finite_series(name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise TrainingStabilityError(f"Keras training history {name!r} must be a non-empty sequence.")
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TrainingStabilityError(f"Keras training history {name!r} must be numeric.") from exc
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise TrainingStabilityError(f"Keras training history {name!r} contains invalid values.")
    return tuple(float(value) for value in numeric)
