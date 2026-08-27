"""Conservative, reproducible training-balance candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class BalanceStrategy(str, Enum):
    """Supported stable-state balancing alternatives."""

    CLASS_WEIGHTS = "class-weights"
    MODERATE_OVERSAMPLING = "moderate-oversampling"
    SYNTHETIC_CONGESTION = "synthetic-congestion"


@dataclass(frozen=True)
class BalanceCandidate:
    """Row positions and supervision weights for one candidate."""

    strategy: BalanceStrategy
    row_positions: np.ndarray
    supervision_weights: np.ndarray


def build_balance_candidates(
    train_frame: pd.DataFrame,
    supervision_weights: np.ndarray,
    *,
    random_state: int,
) -> dict[BalanceStrategy, BalanceCandidate]:
    """Build candidates without modifying validation or test partitions."""
    weights = np.asarray(supervision_weights, dtype=float)
    if len(train_frame) != len(weights):
        raise ValueError("Training rows and supervision weights must have equal length.")
    states = pd.to_numeric(train_frame["traffic_state"], errors="raise").to_numpy(dtype=int)
    if not np.isin(states, (0, 1, 2)).all():
        raise ValueError("Balance candidates support only stable states 0, 1 and 2.")
    synthetic = (
        train_frame.get("data_origin", pd.Series("real", index=train_frame.index))
        .eq("synthetic")
        .to_numpy()
    )
    real_positions = np.flatnonzero(~synthetic)
    if not len(real_positions):
        raise ValueError("At least one real training row is required.")

    candidates = {
        BalanceStrategy.CLASS_WEIGHTS: BalanceCandidate(
            BalanceStrategy.CLASS_WEIGHTS,
            real_positions,
            weights[real_positions],
        ),
        BalanceStrategy.SYNTHETIC_CONGESTION: BalanceCandidate(
            BalanceStrategy.SYNTHETIC_CONGESTION,
            np.arange(len(train_frame), dtype=int),
            weights,
        ),
    }

    rng = np.random.default_rng(random_state)
    oversampled = list(real_positions)
    real_states = states[real_positions]
    normal_support = int((real_states == 0).sum())
    ceiling = max(1, normal_support // 2)
    for state in (1, 2):
        state_positions = real_positions[real_states == state]
        current = len(state_positions)
        if current == 0:
            continue
        target = min(ceiling, current * 4)
        if target > current:
            oversampled.extend(
                rng.choice(state_positions, size=target - current, replace=True).tolist()
            )
    oversampled_positions = np.asarray(oversampled, dtype=int)
    candidates[BalanceStrategy.MODERATE_OVERSAMPLING] = BalanceCandidate(
        BalanceStrategy.MODERATE_OVERSAMPLING,
        oversampled_positions,
        weights[oversampled_positions],
    )
    return candidates


def compute_capped_balanced_weights(
    states: np.ndarray,
    supervision_weights: np.ndarray,
    *,
    maximum_class_weight: float = 4.0,
) -> tuple[np.ndarray, dict[int, float]]:
    """Combine capped class weights with source-supervision weights."""
    target = np.asarray(states, dtype=int)
    supervision = np.asarray(supervision_weights, dtype=float)
    if len(target) != len(supervision):
        raise ValueError("Targets and supervision weights must have equal length.")
    classes = np.unique(target)
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=target)
    class_weights = {
        int(code): min(float(weight), maximum_class_weight)
        for code, weight in zip(classes, balanced, strict=True)
    }
    sample_weights = supervision * np.array(
        [class_weights[int(code)] for code in target], dtype=float
    )
    return sample_weights, class_weights


__all__ = [
    "BalanceCandidate",
    "BalanceStrategy",
    "build_balance_candidates",
    "compute_capped_balanced_weights",
]
