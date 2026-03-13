"""Tests for src/perception/pipeline.py — shared telemetry extraction helpers."""

from __future__ import annotations

from datetime import datetime

from src.perception.pipeline import MinuteTelemetryAccumulator
from src.perception.tracker import Track


class TestMinuteTelemetryAccumulator:
    def test_observe_track_counts_once_and_accumulates_speed(self) -> None:
        accumulator = MinuteTelemetryAccumulator(clip_id="clip_001")
        track = Track(track_id=1, vehicle_type="car", centroid=(100, 100))
        track.history.append((100, 100))

        accumulator.observe_track(
            track,
            smoothed_speed=12.5,
            reliable=True,
            near_zero_motion=False,
            stationary_confirmed=False,
            recovered_gap=0,
        )
        accumulator.observe_track(
            track,
            smoothed_speed=15.0,
            reliable=True,
            near_zero_motion=False,
            stationary_confirmed=False,
            recovered_gap=0,
        )

        record = accumulator.build_record(datetime(2025, 5, 1, 8, 0, 0))
        assert record["count_car"] == 1
        assert record["speed_sample_count"] == 2
        assert record["speed_measurement_quality"] == 1.0

    def test_quality_tracks_rejected_and_stationary_events(self) -> None:
        accumulator = MinuteTelemetryAccumulator(clip_id="clip_001")
        stationary_track = Track(track_id=1, vehicle_type="car", centroid=(100, 100))
        rejected_track = Track(track_id=2, vehicle_type="truck", centroid=(120, 100))
        accepted_track = Track(track_id=3, vehicle_type="bus", centroid=(130, 100))

        for track in (stationary_track, rejected_track, accepted_track):
            track.history.append(track.centroid)

        accumulator.observe_track(
            stationary_track,
            smoothed_speed=None,
            reliable=True,
            near_zero_motion=True,
            stationary_confirmed=True,
            recovered_gap=0,
        )
        accumulator.observe_track(
            rejected_track,
            smoothed_speed=None,
            reliable=False,
            near_zero_motion=True,
            stationary_confirmed=False,
            recovered_gap=1,
        )
        accumulator.observe_track(
            accepted_track,
            smoothed_speed=9.0,
            reliable=True,
            near_zero_motion=False,
            stationary_confirmed=False,
            recovered_gap=0,
        )

        record = accumulator.build_record(datetime(2025, 5, 1, 8, 0, 0))
        assert record["near_zero_motion_count"] == 2
        assert record["stationary_confirmed_count"] == 1
        assert record["rejected_speed_count"] == 1
        assert record["recovered_track_count"] == 1
        assert record["speed_sample_count"] == 1
        assert record["speed_measurement_quality"] == 0.5

    def test_rollover_resets_minute_state_and_keeps_cumulative_counts(self) -> None:
        accumulator = MinuteTelemetryAccumulator(clip_id="clip_001")
        track = Track(track_id=1, vehicle_type="motorcycle", centroid=(50, 50))
        track.history.append((50, 50))

        accumulator.observe_track(
            track,
            smoothed_speed=8.0,
            reliable=True,
            near_zero_motion=False,
            stationary_confirmed=False,
            recovered_gap=0,
        )
        accumulator.rollover_minute()

        assert accumulator.cumulative_counts["motorcycle"] == 1
        assert accumulator.minute_counts["motorcycle"] == 0
        assert accumulator.speed_sample_count == 0
        assert accumulator.near_zero_motion_count == 0
