"""Tests for src/synthetic.py — synthetic edge-case data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaaet.features.synthetic import (
    SYNTHETIC_ID_OFFSET,
    augment_with_synthetic,
    generate_accident_sequences,
    generate_congestion_sequences,
)
from vaaet.settings import (
    DATA_ORIGIN_COL,
    LABELING_THRESHOLDS,
    RANDOM_SEED,
    SYNTHETIC_SCENARIO_COL,
)

_RAW_COLS = {
    "id",
    "clip_id",
    "record_time",
    "avg_speed",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "total_vehicles",
}
_V2_COLS = {
    "near_zero_motion_count",
    "stationary_confirmed_count",
    "rejected_speed_count",
    "recovered_track_count",
    "speed_sample_count",
    "speed_measurement_quality",
    "optical_flow_tracking_ratio",
    "telemetry_schema_version",
}


def _tiny_raw_df(n: int = 5) -> pd.DataFrame:
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


class TestGenerateAccidentSequences:
    def test_row_count(self) -> None:
        df = generate_accident_sequences(n_sequences=3, records_per_seq=8)
        assert len(df) == 3 * 8

    def test_schema_matches_raw(self) -> None:
        df = generate_accident_sequences(n_sequences=1, records_per_seq=5)
        assert _RAW_COLS.issubset(df.columns)
        assert _V2_COLS.issubset(df.columns)

    def test_unique_track_counters_are_bounded(self) -> None:
        df = generate_accident_sequences(n_sequences=2, records_per_seq=10)
        assert (df["near_zero_motion_count"] <= df["total_vehicles"]).all()
        assert (df["stationary_confirmed_count"] <= df["total_vehicles"]).all()
        assert df["speed_measurement_quality"].between(0, 1).all()
        assert df["optical_flow_tracking_ratio"].between(0, 1).all()

    def test_standstill_records_have_low_speed(self) -> None:
        df = generate_accident_sequences(n_sequences=2, records_per_seq=10)
        t = LABELING_THRESHOLDS
        for seq_start in range(0, len(df), 10):
            standstill = df.iloc[seq_start + 5 : seq_start + 10]
            assert (standstill["avg_speed"] <= t["accident_speed_max"]).all()

    def test_ids_start_at_offset(self) -> None:
        df = generate_accident_sequences(n_sequences=1, records_per_seq=5)
        assert df["id"].min() >= SYNTHETIC_ID_OFFSET

    def test_accident_sequences_are_tagged(self) -> None:
        df = generate_accident_sequences(n_sequences=1, records_per_seq=5)
        assert (df[DATA_ORIGIN_COL] == "synthetic").all()
        assert (df[SYNTHETIC_SCENARIO_COL] == "accident").all()

    def test_each_accident_sequence_gets_its_own_clip_id(self) -> None:
        df = generate_accident_sequences(n_sequences=3, records_per_seq=4)
        assert df["clip_id"].nunique() == 3
        assert df["clip_id"].str.startswith("synthetic_accident_").all()


class TestGenerateCongestionSequences:
    def test_row_count(self) -> None:
        df = generate_congestion_sequences(n_sequences=4, records_per_seq=6)
        assert len(df) == 4 * 6

    def test_speed_in_congested_range(self) -> None:
        df = generate_congestion_sequences(n_sequences=3, records_per_seq=10)
        t = LABELING_THRESHOLDS
        assert (df["avg_speed"] > t["accident_speed_max"]).all()
        assert (df["avg_speed"] < t["congested_speed_max"]).all()

    def test_high_vehicle_volume(self) -> None:
        df = generate_congestion_sequences(n_sequences=2, records_per_seq=8)
        t = LABELING_THRESHOLDS
        assert (df["total_vehicles"] > t["congested_vehicles_min"]).all()

    def test_congestion_sequences_are_tagged(self) -> None:
        df = generate_congestion_sequences(n_sequences=1, records_per_seq=5)
        assert (df[DATA_ORIGIN_COL] == "synthetic").all()
        assert (df[SYNTHETIC_SCENARIO_COL] == "congestion").all()
        assert df["clip_id"].str.startswith("synthetic_congestion_").all()


class TestAugmentWithSynthetic:
    def test_output_longer_than_input(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=2, n_congestion_seq=2, records_per_seq=5
        )
        assert len(augmented) == len(raw) + 20

    def test_original_records_preserved(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=1, n_congestion_seq=1, records_per_seq=5
        )
        pd.testing.assert_frame_equal(
            augmented.iloc[: len(raw)][raw.columns].reset_index(drop=True),
            raw,
        )

    def test_real_rows_are_tagged(self) -> None:
        raw = _tiny_raw_df()
        augmented = augment_with_synthetic(
            raw, n_accident_seq=1, n_congestion_seq=1, records_per_seq=5
        )
        real = augmented.iloc[: len(raw)]
        assert (real[DATA_ORIGIN_COL] == "real").all()
        assert (real[SYNTHETIC_SCENARIO_COL] == "observed").all()

    def test_reproducibility_with_seed(self) -> None:
        raw = _tiny_raw_df()
        a = augment_with_synthetic(raw, seed=123)
        b = augment_with_synthetic(raw, seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different_data(self) -> None:
        raw = _tiny_raw_df()
        a = augment_with_synthetic(raw, seed=1)
        b = augment_with_synthetic(raw, seed=2)
        assert not np.allclose(
            a["avg_speed"].values[len(raw) :],
            b["avg_speed"].values[len(raw) :],
        )

    def test_augmented_origin_breakdown_is_explicit(self) -> None:
        raw = _tiny_raw_df(n=3)
        augmented = augment_with_synthetic(
            raw,
            n_accident_seq=1,
            n_congestion_seq=1,
            records_per_seq=2,
            seed=RANDOM_SEED,
        )
        counts = augmented[DATA_ORIGIN_COL].value_counts().to_dict()
        assert counts["real"] == 3
        assert counts["synthetic"] == 4
