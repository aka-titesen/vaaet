"""Tests for src/features.py — feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS
from src.features import engineer_features


class TestEngineerFeatures:
    """Validate the 9 → 14 feature transformation."""

    def test_output_has_14_feature_columns(
        self, raw_telemetry_df: pd.DataFrame
    ) -> None:
        result = engineer_features(raw_telemetry_df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_no_nan_in_features(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        for col in FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_rows_dropped_from_diff(self, raw_telemetry_df: pd.DataFrame) -> None:
        """diff() creates NaN in row 0 — it must be dropped."""
        result = engineer_features(raw_telemetry_df)
        assert len(result) < len(raw_telemetry_df)

    def test_heavy_vehicle_ratio_range(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert (result["heavy_vehicle_ratio"] >= 0).all()
        assert (result["heavy_vehicle_ratio"] <= 1).all()

    def test_transition_flag_binary(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert set(result["transition_flag"].unique()).issubset({0, 1})

    def test_hour_of_day_range(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert (result["hour_of_day"] >= 0).all()
        assert (result["hour_of_day"] <= 23).all()

    def test_weather_condition_binary(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert set(result["weather_condition"].unique()).issubset({0, 1})

    def test_does_not_modify_input(self, raw_telemetry_df: pd.DataFrame) -> None:
        original = raw_telemetry_df.copy()
        engineer_features(raw_telemetry_df)
        pd.testing.assert_frame_equal(raw_telemetry_df, original)

    def test_empty_dataframe(self) -> None:
        """Edge case: empty input should produce empty output."""
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

    def test_speed_variance_non_negative(self, raw_telemetry_df: pd.DataFrame) -> None:
        result = engineer_features(raw_telemetry_df)
        assert (result["speed_variance"] >= 0).all()

    def test_delta_speed_values(self, raw_telemetry_df: pd.DataFrame) -> None:
        """delta_speed should be the first difference of avg_speed."""
        result = engineer_features(raw_telemetry_df)
        # Manually compute expected deltas (after diff + dropna)
        raw_speeds = raw_telemetry_df["avg_speed"].values
        expected_deltas = np.diff(raw_speeds)
        # Result rows correspond to indices 1..N-1 of original
        np.testing.assert_allclose(
            result["delta_speed"].values, expected_deltas, rtol=1e-10
        )
