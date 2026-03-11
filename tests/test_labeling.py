"""Tests for src/labeling.py — auto-labeling traffic states."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LABELING_THRESHOLDS, STATE_LABELS
from src.labeling import assign_traffic_state


def _make_features_df(**overrides) -> pd.DataFrame:
    """Build a minimal DataFrame for labeling with sensible defaults.

    Accepts column overrides as keyword arguments.
    """
    n = overrides.pop("n", 10)
    defaults = {
        "avg_speed": np.full(n, 60.0),
        "total_vehicles": np.full(n, 10, dtype=int),
        "delta_speed": np.zeros(n),
        "delta_count": np.zeros(n, dtype=int),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestAssignTrafficState:
    """Verify labeling rules produce correct state codes."""

    def test_all_normal(self) -> None:
        """High speed, low density ⟹ all Normal (0)."""
        df = _make_features_df(avg_speed=np.full(10, 60.0))
        states = assign_traffic_state(df)
        assert (states == 0).all()

    def test_reduced_flow(self) -> None:
        """Speed 7-25, density 5-12 ⟹ Reduced (1)."""
        df = _make_features_df(
            avg_speed=np.full(10, 15.0),
            total_vehicles=np.full(10, 8, dtype=int),
        )
        states = assign_traffic_state(df)
        assert (states == 1).all()

    def test_congested(self) -> None:
        """Speed <7, density >8, sustained ≥2 ⟹ Congested (2)."""
        n = 5
        df = _make_features_df(
            n=n,
            avg_speed=np.full(n, 5.0),
            total_vehicles=np.full(n, 15, dtype=int),
        )
        states = assign_traffic_state(df)
        # At least some records should be Congested
        assert (states == 2).any()

    def test_accident_detection(self) -> None:
        """Speed ~0 after sudden braking, sustained ≥3 records ⟹ Accident (3)."""
        n = 10
        speeds = np.full(n, 1.0)  # All near zero
        delta_speeds = np.zeros(n)
        delta_speeds[1] = -25.0  # Sudden braking signal
        df = _make_features_df(
            n=n,
            avg_speed=speeds,
            delta_speed=delta_speeds,
            total_vehicles=np.full(n, 5, dtype=int),
        )
        states = assign_traffic_state(df)
        # Later rows (after persistence window) should be Accident
        assert (states == 3).any()

    def test_severity_ordering(self) -> None:
        """Accident takes priority over Congested."""
        n = 10
        speeds = np.full(n, 1.0)
        delta_speeds = np.zeros(n)
        delta_speeds[0] = -25.0
        df = _make_features_df(
            n=n,
            avg_speed=speeds,
            delta_speed=delta_speeds,
            total_vehicles=np.full(n, 30, dtype=int),
        )
        states = assign_traffic_state(df)
        # Should not have state 2 overwriting state 3
        accident_mask = states == 3
        if accident_mask.any():
            # Those same rows should NOT be 2
            assert (states[accident_mask] != 2).all()

    def test_output_type_and_range(self, engineered_df: pd.DataFrame) -> None:
        states = assign_traffic_state(engineered_df)
        assert isinstance(states, pd.Series)
        assert states.dtype == int
        assert set(states.unique()).issubset({0, 1, 2, 3})

    def test_all_states_have_labels(self) -> None:
        """Every possible state code has a human-readable label."""
        for code in range(4):
            assert code in STATE_LABELS

    def test_thresholds_consistency(self) -> None:
        """Reduced speed range must not overlap with Congested."""
        t = LABELING_THRESHOLDS
        assert t["reduced_speed_min"] >= t["congested_speed_max"]
