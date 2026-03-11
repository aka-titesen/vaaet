"""Tests for src/synthetic.py — synthetic edge-case data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LABELING_THRESHOLDS, RANDOM_SEED
from src.synthetic import (
    SYNTHETIC_ID_OFFSET,
    augment_with_synthetic,
    generate_accident_sequences,
    generate_congestion_sequences,
)

# Required columns matching the traffic_data schema
_RAW_COLS = {
    "id",
    "record_time",
    "avg_speed",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "total_vehicles",
}


# Helpers


def _tiny_raw_df(n: int = 5) -> pd.DataFrame:
    """Minimal real-telemetry-like DataFrame used as augment input."""
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "record_time": pd.date_range("2025-05-01", periods=n, freq="1min"),
            "avg_speed": np.full(n, 50.0),
            "count_car": np.full(n, 5, dtype=int),
            "count_truck": np.full(n, 1, dtype=int),
            "count_bus": np.full(n, 0, dtype=int),
            "count_motorcycle": np.full(n, 0, dtype=int),
            "count_bicycle": np.full(n, 0, dtype=int),
            "total_vehicles": np.full(n, 6, dtype=int),
        }
    )


# Accident sequences


class TestGenerateAccidentSequences:
    """Verify accident-sequence generator produces plausible data."""

    def test_row_count(self) -> None:
        df = generate_accident_sequences(n_sequences=3, records_per_seq=8)
        assert len(df) == 3 * 8

    def test_schema_matches_raw(self) -> None:
        df = generate_accident_sequences(n_sequences=1, records_per_seq=5)
        assert _RAW_COLS.issubset(df.columns)

    def test_standstill_records_have_low_speed(self) -> None:
        """Last ~50 % of each sequence should have speed ≤ accident_speed_max."""
        df = generate_accident_sequences(n_sequences=2, records_per_seq=10)
        t = LABELING_THRESHOLDS
        # Standstill starts after index 4 within each 10-record sequence
        for seq_start in range(0, len(df), 10):
            standstill = df.iloc[seq_start + 5 : seq_start + 10]
            assert (standstill["avg_speed"] <= t["accident_speed_max"]).all()

    def test_ids_start_at_offset(self) -> None:
        df = generate_accident_sequences(n_sequences=1, records_per_seq=5)
        assert df["id"].min() >= SYNTHETIC_ID_OFFSET

    def test_total_vehicles_positive(self) -> None:
        df = generate_accident_sequences(n_sequences=2, records_per_seq=8)
        assert (df["total_vehicles"] >= 1).all()


# Congestion sequences


class TestGenerateCongestionSequences:
    """Verify congestion-sequence generator produces plausible data."""

    def test_row_count(self) -> None:
        df = generate_congestion_sequences(n_sequences=4, records_per_seq=6)
        assert len(df) == 4 * 6

    def test_schema_matches_raw(self) -> None:
        df = generate_congestion_sequences(n_sequences=1, records_per_seq=5)
        assert _RAW_COLS.issubset(df.columns)

    def test_speed_in_congested_range(self) -> None:
        """Speed should be above accident threshold but below congested ceiling."""
        df = generate_congestion_sequences(n_sequences=3, records_per_seq=10)
        t = LABELING_THRESHOLDS
        assert (df["avg_speed"] > t["accident_speed_max"]).all()
        assert (df["avg_speed"] < t["congested_speed_max"]).all()

    def test_high_vehicle_volume(self) -> None:
        """Every record should exceed congested_vehicles_min."""
        df = generate_congestion_sequences(n_sequences=2, records_per_seq=8)
        t = LABELING_THRESHOLDS
        assert (df["total_vehicles"] > t["congested_vehicles_min"]).all()

    def test_ids_distinguishable(self) -> None:
        df = generate_congestion_sequences(n_sequences=1, records_per_seq=5)
        assert df["id"].min() >= SYNTHETIC_ID_OFFSET


# augment_with_synthetic


class TestAugmentWithSynthetic:
    """Verify the public augment API."""

    def test_output_longer_than_input(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=2, n_congestion_seq=2, records_per_seq=5
        )
        assert len(augmented) > len(raw)
        assert len(augmented) == len(raw) + 2 * 5 + 2 * 5

    def test_original_records_preserved(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=1, n_congestion_seq=1, records_per_seq=5
        )
        # First rows should be original
        pd.testing.assert_frame_equal(
            augmented.iloc[: len(raw)].reset_index(drop=True), raw
        )

    def test_synthetic_ids_above_offset(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=1, n_congestion_seq=1, records_per_seq=5
        )
        synthetic = augmented.iloc[len(raw) :]
        assert (synthetic["id"] >= SYNTHETIC_ID_OFFSET).all()

    def test_reproducibility_with_seed(self) -> None:
        raw = _tiny_raw_df()
        a = augment_with_synthetic(raw, seed=123)
        b = augment_with_synthetic(raw, seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different_data(self) -> None:
        raw = _tiny_raw_df()
        a = augment_with_synthetic(raw, seed=1)
        b = augment_with_synthetic(raw, seed=2)
        # At least speeds should differ
        assert not np.allclose(
            a["avg_speed"].values[len(raw) :],
            b["avg_speed"].values[len(raw) :],
        )
