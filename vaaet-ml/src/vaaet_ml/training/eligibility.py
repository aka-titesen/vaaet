# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Gates actuales de elegibilidad de candidatos, separados del notebook."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from vaaet_ml.data.datasets import build_group_ids
from vaaet_ml.training.lifecycle import TrainingMode

__all__ = ["CandidateEligibility", "evaluate_candidate_eligibility"]

_MINIMUM_F1_MACRO = 0.88
_MINIMUM_PRECISION = MappingProxyType({0: 0.93, 1: 0.88, 2: 0.90})
_MINIMUM_RECALL = MappingProxyType({0: 0.93, 1: 0.90, 2: 0.85})
_MAXIMUM_DIRECT_NORMAL_CONGESTED_ERROR = 0.01
_MAXIMUM_ECE = 0.05
_MINIMUM_CONGESTED_MINUTES = 100
_MINIMUM_CONGESTED_CLIPS = 20
_MINIMUM_NEGATIVE_EXPOSURE_HOURS = 300.0
_MAXIMUM_FALSE_CANDIDATES_PER_HOUR = 0.01


@dataclass(frozen=True)
class CandidateEligibility:
    """Resultado explicable de los gates vigentes; no ejecuta una promoción."""

    metric_gates: dict[str, bool]
    promotion_blockers: tuple[str, ...]
    human_holdout: bool
    congested_minutes: int
    congested_clips: int
    production_eligible: bool


def evaluate_candidate_eligibility(
    *,
    training_mode: TrainingMode | str,
    dataset_blockers: Sequence[str],
    human_holdout: bool,
    test_frame: pd.DataFrame,
    actual: Sequence[int],
    predicted: Sequence[int],
    f1_macro: float,
    direct_normal_congested_error: float,
    expected_calibration_error: float,
    negative_exposure_hours: float,
    false_candidates_per_hour: float,
) -> CandidateEligibility:
    """Aplica los mismos umbrales del notebook sin crear criterios nuevos."""

    mode = TrainingMode(training_mode)
    truth = np.asarray(actual, dtype=int)
    final = np.asarray(predicted, dtype=int)
    if truth.ndim != 1 or truth.shape != final.shape or len(truth) != len(test_frame):
        raise ValueError("Eligibility requires aligned one-dimensional test targets and predictions.")
    if not np.isin(truth, (0, 1, 2)).all() or not np.isin(final, (0, 1, 2)).all():
        raise ValueError("Eligibility is defined only for the three stable model states.")
    _validate_metric("f1_macro", f1_macro)
    _validate_metric("direct_normal_congested_error", direct_normal_congested_error)
    _validate_metric("expected_calibration_error", expected_calibration_error)
    _validate_metric("negative_exposure_hours", negative_exposure_hours)
    if negative_exposure_hours > 0:
        _validate_metric("false_candidates_per_hour", false_candidates_per_hour)

    classification = classification_report(
        truth,
        final,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0,
    )
    metric_gates = {
        "f1_macro": f1_macro >= _MINIMUM_F1_MACRO,
        **{
            f"{label}_precision": classification[str(state)]["precision"]
            >= _MINIMUM_PRECISION[state]
            for state, label in ((0, "normal"), (1, "reduced"), (2, "congested"))
        },
        **{
            f"{label}_recall": classification[str(state)]["recall"] >= _MINIMUM_RECALL[state]
            for state, label in ((0, "normal"), (1, "reduced"), (2, "congested"))
        },
        "direct_normal_congested_error": (
            direct_normal_congested_error <= _MAXIMUM_DIRECT_NORMAL_CONGESTED_ERROR
        ),
        "ece": expected_calibration_error <= _MAXIMUM_ECE,
    }
    blockers = list(dataset_blockers)
    if mode is TrainingMode.SEED_BOOTSTRAP:
        blockers.append("seed bootstrap uses weak proxy supervision and is pilot-only")
    if not human_holdout:
        blockers.append("validation/test are not a frozen human-validated holdout")
    blockers.extend(f"metric gate failed: {name}" for name, passed in metric_gates.items() if not passed)

    congested_minutes = int((truth == 2).sum())
    congested_clips = _congested_clip_count(test_frame, truth)
    if (
        congested_minutes < _MINIMUM_CONGESTED_MINUTES
        or congested_clips < _MINIMUM_CONGESTED_CLIPS
    ):
        blockers.append(
            f"Congested support insufficient: {congested_minutes} minutes / {congested_clips} clips"
        )
    if negative_exposure_hours < _MINIMUM_NEGATIVE_EXPOSURE_HOURS:
        blockers.append(
            f"incident negative exposure insufficient: {negative_exposure_hours:.2f}/300 h"
        )
    elif false_candidates_per_hour >= _MAXIMUM_FALSE_CANDIDATES_PER_HOUR:
        blockers.append("incident candidate rate is not below 1 per 100 hours")
    return CandidateEligibility(
        metric_gates=metric_gates,
        promotion_blockers=tuple(blockers),
        human_holdout=human_holdout,
        congested_minutes=congested_minutes,
        congested_clips=congested_clips,
        production_eligible=not blockers and mode is TrainingMode.HITL_RETRAINING,
    )


def _congested_clip_count(test_frame: pd.DataFrame, truth: np.ndarray) -> int:
    congested = test_frame.iloc[np.flatnonzero(truth == 2)]
    return 0 if congested.empty else int(build_group_ids(congested).nunique())


def _validate_metric(name: str, value: float) -> None:
    if not isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative value.")
