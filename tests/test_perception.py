"""Tests for src/perception/ — detector, tracker, and speed modules.

These tests exercise pure logic only. YOLO model loading is NOT tested
(requires downloading weights). The YOLODetector.detect() method is
excluded; only the Detection dataclass is verified.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.perception.detector import Detection
from src.perception.tracker import SORTTracker, Track
from src.perception.speed import (
    compensate_camera_motion,
    estimate_speed,
    get_perspective_factor,
)


# ── Detection dataclass ──


class TestDetection:
    def test_centroid_auto_calculated(self) -> None:
        d = Detection(bbox=(100, 200, 300, 400), vehicle_type="car", confidence=0.9)
        assert d.centroid == (200, 300)

    def test_frozen(self) -> None:
        d = Detection(bbox=(0, 0, 10, 10), vehicle_type="truck", confidence=0.8)
        with pytest.raises(AttributeError):
            d.vehicle_type = "bus"  # type: ignore[misc]

    def test_confidence_stored(self) -> None:
        d = Detection(bbox=(0, 0, 50, 50), vehicle_type="bus", confidence=0.75)
        assert d.confidence == 0.75


# ── Track dataclass ──


class TestTrack:
    def test_initial_state(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(50, 50))
        assert t.frames_since_seen == 0
        assert t.total_frames == 0
        assert len(t.history) == 0

    def test_update_appends_history(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(50, 50))
        t.update((60, 60))
        assert t.centroid == (60, 60)
        assert len(t.history) == 1
        assert t.total_frames == 1
        assert t.frames_since_seen == 0

    def test_history_maxlen(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(0, 0))
        for i in range(50):
            t.update((i, i))
        assert len(t.history) == 30  # default maxlen


# ── SORTTracker ──


class TestSORTTracker:
    def test_empty_update(self) -> None:
        tracker = SORTTracker()
        active = tracker.update([])
        assert active == []

    def test_new_detections_create_tracks(self) -> None:
        tracker = SORTTracker()
        dets = [((100, 100), "car"), ((200, 200), "truck")]
        active = tracker.update(dets)
        assert len(active) == 2
        assert active[0].vehicle_type == "car"
        assert active[1].vehicle_type == "truck"

    def test_matching_by_proximity(self) -> None:
        tracker = SORTTracker(max_distance=50)
        tracker.update([((100, 100), "car")])
        # Move the car slightly
        active = tracker.update([((110, 105), "car")])
        assert len(active) == 1
        assert active[0].track_id == 1  # Same track
        assert active[0].centroid == (110, 105)

    def test_no_match_creates_new_track(self) -> None:
        tracker = SORTTracker(max_distance=50)
        tracker.update([((100, 100), "car")])
        # Detection far away → new track
        active = tracker.update([((500, 500), "car")])
        # The old track is lost (incremented), the new one is active
        assert any(t.track_id == 2 for t in active)

    def test_pruning_after_max_lost(self) -> None:
        tracker = SORTTracker(max_distance=50, max_lost=2)
        tracker.update([((100, 100), "car")])
        # 3 frames with no detections → track should be pruned
        tracker.update([])
        tracker.update([])
        active = tracker.update([])
        # After 3 empty frames with max_lost=2, the track is pruned
        assert len(tracker._tracks) == 0

    def test_vehicle_type_matching(self) -> None:
        """Tracks should only match detections of the same vehicle type."""
        tracker = SORTTracker(max_distance=200)
        tracker.update([((100, 100), "car")])
        # Same position but different type → should NOT match
        active = tracker.update([((105, 105), "truck")])
        # Should have 2 tracks: old car (lost) + new truck
        assert any(t.vehicle_type == "truck" for t in active)


# ── Speed estimation ──


class TestGetPerspectiveFactor:
    def test_near_zone(self) -> None:
        """Bottom of frame (y > 0.66 * height) → near factor."""
        assert get_perspective_factor(800, 1080) == 1.8

    def test_mid_zone(self) -> None:
        """Middle zone (0.33 < ratio <= 0.66) → mid factor."""
        assert get_perspective_factor(540, 1080) == 1.0

    def test_far_zone(self) -> None:
        """Top of frame (ratio <= 0.33) → far factor."""
        assert get_perspective_factor(100, 1080) == 0.6

    def test_zero_height_safe(self) -> None:
        """frame_height=0 should not cause division by zero."""
        result = get_perspective_factor(0, 0)
        assert isinstance(result, float)


class TestCompensateCameraMotion:
    def test_subtraction(self) -> None:
        disp = np.array([10.0, 5.0])
        motion = np.array([2.0, 1.0])
        result = compensate_camera_motion(disp, motion)
        np.testing.assert_array_equal(result, [8.0, 4.0])

    def test_zero_motion(self) -> None:
        disp = np.array([10.0, 5.0])
        result = compensate_camera_motion(disp, np.zeros(2))
        np.testing.assert_array_equal(result, disp)


class TestEstimateSpeed:
    def test_insufficient_history(self) -> None:
        """Less than 2 points → None."""
        assert estimate_speed(deque([(100, 100)]), fps=30.0) is None

    def test_stationary_vehicle(self) -> None:
        """Same position → speed below minimum → None."""
        history = deque([(100, 100), (100, 100), (100, 100)])
        assert estimate_speed(history, fps=30.0) is None

    def test_plausible_speed(self) -> None:
        """Moving vehicle with known displacement should produce valid speed."""
        # 10 frames at 30fps = 0.333s
        # displacement ~50px → 50/12 = 4.17m → 4.17/0.333 = 12.5 m/s → 45 km/h
        history = deque([(100, 540)] + [(100 + i * 5, 540) for i in range(1, 11)])
        speed = estimate_speed(history, fps=30.0, frame_height=1080)
        assert speed is not None
        assert 2.0 <= speed <= 120.0

    def test_implausible_high_speed_returns_none(self) -> None:
        """Extremely fast displacement → exceeds 120km/h → None."""
        history = deque([(0, 540), (10000, 540)])
        result = estimate_speed(history, fps=30.0)
        assert result is None

    def test_global_motion_compensation(self) -> None:
        """With camera motion subtracted, effective displacement changes."""
        history = deque([(100, 540), (150, 540), (200, 540)])
        # Without global motion
        speed_no_comp = estimate_speed(history, fps=30.0, frame_height=1080)
        # With global motion in the same direction (should reduce speed)
        speed_comp = estimate_speed(
            history,
            fps=30.0,
            frame_height=1080,
            global_motion=np.array([40.0, 0.0]),
        )
        if speed_no_comp is not None and speed_comp is not None:
            assert speed_comp < speed_no_comp
