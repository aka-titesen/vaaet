"""Tests for src/calibration.py — lightweight speed validation helpers."""

from __future__ import annotations

import pytest

from vaaet.evaluation.calibration import (
    CalibrationSegment,
    aggregate_pixels_per_meter,
    build_calibration_table,
    pixels_per_meter_from_segment,
    pseudo_ground_truth_speed_kmh,
)


class TestCalibrationHelpers:
    def test_pixels_per_meter_from_segment(self) -> None:
        segment = CalibrationSegment(
            name="lane_mark",
            pixel_start=(0.0, 0.0),
            pixel_end=(120.0, 0.0),
            meters=10.0,
        )
        assert pixels_per_meter_from_segment(segment) == 12.0

    def test_aggregate_pixels_per_meter_uses_median(self) -> None:
        segments = [
            CalibrationSegment("a", (0.0, 0.0), (120.0, 0.0), meters=10.0),
            CalibrationSegment("b", (0.0, 0.0), (132.0, 0.0), meters=11.0),
            CalibrationSegment("c", (0.0, 0.0), (300.0, 0.0), meters=10.0),
        ]
        assert aggregate_pixels_per_meter(segments) == 12.0

    def test_pseudo_ground_truth_speed_kmh(self) -> None:
        assert pseudo_ground_truth_speed_kmh(distance_m=100.0, elapsed_seconds=10.0) == 36.0

    def test_build_calibration_table_includes_optional_speed(self) -> None:
        segments = [
            CalibrationSegment(
                name="bridge_segment",
                pixel_start=(0.0, 0.0),
                pixel_end=(120.0, 0.0),
                meters=10.0,
                elapsed_seconds=2.0,
            )
        ]
        table = build_calibration_table(segments)
        assert table.loc[0, "pixels_per_meter"] == 12.0
        assert table.loc[0, "pseudo_ground_truth_speed_kmh"] == 18.0

    def test_invalid_distance_raises(self) -> None:
        with pytest.raises(ValueError):
            pseudo_ground_truth_speed_kmh(distance_m=0.0, elapsed_seconds=1.0)
