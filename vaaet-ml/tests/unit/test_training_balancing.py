from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaaet_ml.training.balancing import (
    BalanceStrategy,
    build_balance_candidates,
    compute_capped_balanced_weights,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "traffic_state": [0] * 12 + [1] * 3 + [2] * 2 + [2] * 4,
            "data_origin": ["real"] * 17 + ["synthetic"] * 4,
        }
    )


def test_balance_candidates_keep_synthetic_only_in_synthetic_strategy() -> None:
    frame = _training_frame()
    candidates = build_balance_candidates(
        frame, np.ones(len(frame)), random_state=42
    )

    class_weight_rows = candidates[BalanceStrategy.CLASS_WEIGHTS].row_positions
    oversampled_rows = candidates[BalanceStrategy.MODERATE_OVERSAMPLING].row_positions
    synthetic_rows = candidates[BalanceStrategy.SYNTHETIC_CONGESTION].row_positions

    assert len(class_weight_rows) == 17
    assert (class_weight_rows < 17).all()
    assert (oversampled_rows < 17).all()
    assert len(oversampled_rows) > len(class_weight_rows)
    assert len(synthetic_rows) == len(frame)


def test_moderate_oversampling_never_exceeds_half_normal_support() -> None:
    frame = _training_frame()
    candidate = build_balance_candidates(
        frame, np.ones(len(frame)), random_state=42
    )[BalanceStrategy.MODERATE_OVERSAMPLING]
    target = frame.iloc[candidate.row_positions]["traffic_state"]
    normal = int(target.eq(0).sum())
    assert int(target.eq(1).sum()) <= normal // 2
    assert int(target.eq(2).sum()) <= normal // 2


def test_balance_candidates_are_reproducible() -> None:
    frame = _training_frame()
    first = build_balance_candidates(frame, np.ones(len(frame)), random_state=7)
    second = build_balance_candidates(frame, np.ones(len(frame)), random_state=7)
    np.testing.assert_array_equal(
        first[BalanceStrategy.MODERATE_OVERSAMPLING].row_positions,
        second[BalanceStrategy.MODERATE_OVERSAMPLING].row_positions,
    )


def test_balanced_weights_are_capped_and_preserve_supervision() -> None:
    states = np.array([0] * 10 + [1] * 2 + [2])
    supervision = np.ones(len(states))
    supervision[-1] = 0.25
    sample, class_weights = compute_capped_balanced_weights(states, supervision)

    assert max(class_weights.values()) <= 4.0
    assert sample[-1] == pytest.approx(class_weights[2] * 0.25)


def test_balance_candidates_reject_accident() -> None:
    frame = _training_frame()
    frame.loc[0, "traffic_state"] = 3
    with pytest.raises(ValueError, match="stable states"):
        build_balance_candidates(frame, np.ones(len(frame)), random_state=42)
