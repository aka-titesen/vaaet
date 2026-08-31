# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contratos del ciclo de ajuste Keras reusable del laboratorio."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vaaet_ml.exceptions import TrainingStabilityError
from vaaet_ml.training.execution import (
    TrainingFitConfig,
    build_training_callbacks,
    reset_training_state,
    validate_training_history,
    validate_training_inputs,
)


def _history(**series: list[float]) -> SimpleNamespace:
    defaults = {
        "loss": [0.5, 0.4],
        "val_loss": [0.6, 0.45],
        "accuracy": [0.7, 0.8],
        "val_accuracy": [0.65, 0.75],
    }
    defaults.update(series)
    return SimpleNamespace(history=defaults)


def test_training_callbacks_are_fresh_and_include_stability_controls() -> None:
    first = build_training_callbacks(TrainingFitConfig(random_seed=42))
    second = build_training_callbacks(TrainingFitConfig(random_seed=42))

    assert [type(callback).__name__ for callback in first] == [
        "EarlyStopping",
        "ReduceLROnPlateau",
        "TerminateOnNaN",
    ]
    assert all(left is not right for left, right in zip(first, second, strict=True))


def test_reset_training_state_clears_then_seeds() -> None:
    events: list[object] = []

    reset_training_state(
        TrainingFitConfig(random_seed=7),
        clear_session=lambda: events.append("cleared"),
        set_random_seed=events.append,
    )

    assert events == ["cleared", 7]


def test_training_inputs_reject_non_finite_features() -> None:
    values = np.ones((2, 3))
    values[0, 0] = np.nan

    with pytest.raises(TrainingStabilityError, match="NaN or infinite"):
        validate_training_inputs(
            values,
            np.array([0, 1]),
            np.ones((2, 3)),
            np.array([0, 1]),
            input_features=3,
            output_classes=3,
        )


def test_training_inputs_reject_out_of_contract_labels() -> None:
    with pytest.raises(TrainingStabilityError, match="state codes"):
        validate_training_inputs(
            np.ones((2, 3)),
            np.array([0, 3]),
            np.ones((2, 3)),
            np.array([0, 1]),
            input_features=3,
            output_classes=3,
        )


def test_training_history_requires_complete_finite_series() -> None:
    result = validate_training_history(_history())
    assert result["val_accuracy"] == (0.65, 0.75)

    with pytest.raises(TrainingStabilityError, match="invalid values"):
        validate_training_history(_history(val_loss=[float("nan")]))
