# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vaaet_ml.evaluation.reporting import (  # noqa: E402
    plot_training_evaluation,
    plot_training_history,
    show_inference_dashboard,
)


def test_plot_training_history_requires_complete_history() -> None:
    with pytest.raises(ValueError, match="missing required series"):
        plot_training_history({"loss": [1.0]})


def test_plot_training_history_and_evaluation_accept_valid_inputs() -> None:
    plot_training_history(
        {"loss": [1.0], "val_loss": [1.1], "accuracy": [0.7], "val_accuracy": [0.6]}
    )
    plot_training_evaluation(
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    )


def test_plot_training_evaluation_validates_probability_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_training_evaluation([0], [0], [0], np.array([[1.0, 0.0]]))


def test_show_inference_dashboard_accepts_canonical_minute_frame() -> None:
    frame = pd.DataFrame(
        {
            "traffic_state": [0, 1],
            "avg_speed": [30.0, 18.0],
            "confidence": [0.95, 0.82],
            "total_vehicles": [4, 7],
            "count_car": [3, 5],
        }
    )

    show_inference_dashboard(frame)


def test_show_inference_dashboard_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="at least one classified minute"):
        show_inference_dashboard(
            pd.DataFrame(columns=["traffic_state", "avg_speed", "confidence"])
        )
