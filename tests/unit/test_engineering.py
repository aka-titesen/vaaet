"""Tests for src/features.py — feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_ORIGIN_COL, FEATURE_COLS, SYNTHETIC_SCENARIO_COL
from src.features import engineer_features
from src.synthetic import augment_with_synthetic


class TestEngineerFeatures:
    """Validate the raw telemetry → engineered feature transformation."""

    def test_output_has_feature_columns(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_no_nan_in_features(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        for col in FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_rows_dropped_from_diff(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert len(result) < len(raw_telemetry_df)

    def test_quality_signals_default_conservatively(
        self, raw_telemetry_df: pd.DataFrame
    ) -> None:
        result = engineer_features(raw_telemetry_df)
        assert (result["speed_measurement_quality"] == 1.0).all()
        assert (result["near_zero_motion_ratio"] == 0.0).all()
        assert (result["stationary_confirmed_ratio"] == 0.0).all()

    def test_quality_signals_are_derived_when_counts_exist(self) -> None:
        df = pd.DataFrame(
            {
                "record_time": pd.date_range("2025-05-01", periods=3, freq="1min"),
                "avg_speed": [20.0, 10.0, 2.0],
                "total_vehicles": [5, 5, 5],
                "count_car": [3, 3, 3],
                "count_truck": [1, 1, 1],
                "count_bus": [0, 0, 0],
                "count_motorcycle": [1, 1, 1],
                "count_bicycle": [0, 0, 0],
                "speed_sample_count": [4, 4, 4],
                "rejected_speed_count": [0, 2, 1],
                "near_zero_motion_count": [0, 1, 3],
                "stationary_confirmed_count": [0, 0, 2],
            }
        )
        result = engineer_features(df)
        assert result["speed_measurement_quality"].iloc[-1] == 0.8
        assert result["near_zero_motion_ratio"].iloc[-1] == 0.75
        assert result["stationary_confirmed_ratio"].iloc[-1] == 0.5

    def test_metadata_columns_survive_feature_engineering(
        self, raw_telemetry_df: pd.DataFrame
    ) -> None:
        augmented = augment_with_synthetic(
            raw_telemetry_df.head(5),
            n_accident_seq=1,
            n_congestion_seq=1,
            records_per_seq=4,
            seed=123,
        )
        result = engineer_features(augmented)
        assert DATA_ORIGIN_COL in result.columns
        assert SYNTHETIC_SCENARIO_COL in result.columns
        assert result[DATA_ORIGIN_COL].isin({"real", "synthetic"}).all()
        assert result[SYNTHETIC_SCENARIO_COL].isin(
            {"observed", "accident", "congestion"}
        ).all()
        assert "clip_id" in result.columns

    def test_does_not_modify_input(self, raw_telemetry_df: pd.DataFrame) -> None:
        original = raw_telemetry_df.copy()
        engineer_features(raw_telemetry_df)
        pd.testing.assert_frame_equal(raw_telemetry_df, original)

    def test_empty_dataframe(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "avg_speed",
                "total_vehicles",
                "count_truck",
                "count_bus",
                "record_time",
                "count_car",
                "count_motorcycle",
                "count_bicycle",
            ]
        )
        result = engineer_features(empty)
        assert len(result) == 0
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_delta_speed_values(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        raw_speeds = raw_telemetry_df["avg_speed"].values
        expected_deltas = np.diff(raw_speeds)
        np.testing.assert_allclose(
            result["delta_speed"].values, expected_deltas, rtol=1e-10
        )

