from __future__ import annotations

import numpy as np

from vaaet_ml.training.selection import _macro_f1


def test_macro_f1_is_perfect_for_stable_state_matches() -> None:
    actual = np.array([0, 1, 2])

    assert _macro_f1(actual, actual) == 1.0
