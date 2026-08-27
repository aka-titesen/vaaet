"""Tests for vaaet.vision detector, tracker, and speed modules.

These tests exercise pure logic only. YOLO model loading is NOT tested
(requires downloading weights). The YOLODetector.detect() method is
excluded; only the Detection dataclass is verified.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from vaaet.vision.detector import Detection, select_model_variant
from vaaet.vision.optical_flow import OpticalFlowEstimator
from vaaet.vision.speed import (
    SmoothedSpeedTracker,
    TrackMotionStateTracker,
    compensate_camera_motion,
    estimate_speed,
    fuse_speed,
    get_perspective_factor,
    is_near_zero_motion,
    is_speed_measurement_reliable,
    is_stationary,
    robust_speed_summary,
)
from vaaet.vision.tracking import SORTTracker, Track

# Detection dataclass


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


# Track dataclass


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
        for i in range(60):
            t.update((i, i))
        assert len(t.history) == 50  # TRACKER_HISTORY_MAXLEN


# SORTTracker


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
        tracker.update([])
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

    def test_recovered_track_exposes_gap(self) -> None:
        tracker = SORTTracker(max_distance=50, max_lost=5)
        tracker.update([((100, 100), "car")])
        tracker.update([])
        active = tracker.update([((110, 100), "car")])
        assert len(active) == 1
        assert active[0].track_id == 1
        assert active[0].recovered_after_gap == 1

    def test_prune_tracks_reports_removed_ids(self) -> None:
        tracker = SORTTracker(max_distance=50, max_lost=1)
        tracker.update([((100, 100), "car")])
        tracker.update([])
        tracker.update([])
        assert tracker.last_pruned_track_ids == [1]


# Speed estimation


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

    def test_near_boundary_is_blended(self) -> None:
        """Perspective factor should change smoothly around the near threshold."""
        result = get_perspective_factor(int(0.66 * 1080), 1080)
        assert 1.0 < result < 1.8

    def test_mid_boundary_is_blended(self) -> None:
        """Perspective factor should change smoothly around the mid threshold."""
        result = get_perspective_factor(int(0.33 * 1080), 1080)
        assert 0.6 < result < 1.0


class TestOpticalFlowEstimator:
    def test_low_tracking_ratio_returns_zero_motion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        estimator = OpticalFlowEstimator(
            grid_step=10, border_margin=0, min_tracking_ratio=0.75
        )
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        estimator._prev_gray = np.zeros((40, 40), dtype=np.uint8)

        def fake_lk(prev_gray, gray, pts, _next_pts, **kwargs):
            new_pts = pts.copy()
            status = np.zeros((len(pts), 1), dtype=np.uint8)
            status[:2] = 1
            return new_pts, status, None

        monkeypatch.setattr(
            "vaaet.vision.optical_flow.cv2.calcOpticalFlowPyrLK", fake_lk
        )

        motion = estimator.update(frame)
        np.testing.assert_array_equal(motion, np.zeros(2))
        assert estimator.last_tracking_ratio < 0.75

    def test_high_tracking_ratio_returns_median_motion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        estimator = OpticalFlowEstimator(
            grid_step=10, border_margin=0, min_tracking_ratio=0.1
        )
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        estimator._prev_gray = np.zeros((40, 40), dtype=np.uint8)

        def fake_lk(prev_gray, gray, pts, _next_pts, **kwargs):
            new_pts = pts + np.array([2.0, 1.0], dtype=np.float32)
            status = np.ones((len(pts), 1), dtype=np.uint8)
            return new_pts, status, None

        monkeypatch.setattr(
            "vaaet.vision.optical_flow.cv2.calcOpticalFlowPyrLK", fake_lk
        )

        motion = estimator.update(frame)
        np.testing.assert_allclose(motion, np.array([2.0, 1.0]))
        assert estimator.last_tracking_ratio == pytest.approx(1.0)


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
        # 25 frames at 30fps (>= SPEED_MIN_TRACK_LENGTH=8)
        # displacement ~120px → 120/12 = 10m → 10/0.8 = 12.5m/s → 45 km/h
        history = deque([(100 + i * 5, 540) for i in range(25)])
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

    def test_recovered_track_speed_is_not_reliable(self) -> None:
        history = deque([(100 + i * 5, 540) for i in range(15)])
        reliable = is_speed_measurement_reliable(
            history,
            flow_tracking_ratio=1.0,
            recovered_after_gap=1,
        )
        assert reliable is False

    def test_low_flow_confidence_speed_is_not_reliable(self) -> None:
        history = deque([(100 + i * 5, 540) for i in range(15)])
        reliable = is_speed_measurement_reliable(
            history,
            flow_tracking_ratio=0.1,
            recovered_after_gap=0,
        )
        assert reliable is False

    def test_last_jump_speed_is_not_reliable(self) -> None:
        history = deque(
            [
                (100, 540),
                (105, 540),
                (110, 540),
                (115, 540),
                (250, 540),
                (255, 540),
                (260, 540),
                (265, 540),
            ],
        )
        reliable = is_speed_measurement_reliable(
            history,
            flow_tracking_ratio=1.0,
            recovered_after_gap=0,
        )
        assert reliable is False


# Track.mark_counted


class TestTrackCounted:
    def test_initial_not_counted(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(50, 50))
        assert t.counted is False

    def test_mark_counted_returns_true_first_call(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(50, 50))
        assert t.mark_counted() is True
        assert t.counted is True

    def test_mark_counted_returns_false_second_call(self) -> None:
        t = Track(track_id=1, vehicle_type="car", centroid=(50, 50))
        t.mark_counted()
        assert t.mark_counted() is False


# SORTTracker extras


class TestSORTTrackerExtras:
    def test_reset_clears_all_tracks(self) -> None:
        tracker = SORTTracker()
        tracker.update([((100, 100), "car")])
        assert len(tracker.all_tracks) == 1
        tracker.reset()
        assert len(tracker.all_tracks) == 0
        assert tracker._next_id == 1

    def test_all_tracks_includes_lost(self) -> None:
        tracker = SORTTracker(max_distance=50, max_lost=5)
        tracker.update([((100, 100), "car")])
        tracker.update([])  # car is now lost
        assert len(tracker.active_tracks) == 0
        assert len(tracker.all_tracks) == 1


# is_stationary


class TestIsStationary:
    def test_single_point_is_stationary(self) -> None:
        assert is_stationary(deque([(100, 100)])) is True

    def test_identical_positions_is_stationary(self) -> None:
        history = deque([(100, 100)] * 20)
        assert is_stationary(history) is True

    def test_small_jitter_is_stationary(self) -> None:
        """Tiny sub-pixel jitter should still be stationary."""
        history = deque([(100 + (i % 2) * 0.1, 100) for i in range(20)])
        assert is_stationary(history) is True

    def test_moving_vehicle_is_not_stationary(self) -> None:
        """Clear movement should not be stationary."""
        history = deque([(100 + i * 10, 100) for i in range(20)])
        assert is_stationary(history) is False


# fuse_speed


class TestFuseSpeed:
    def test_physics_only_when_mlp_none(self) -> None:
        assert fuse_speed(50.0, None) == 50.0

    def test_fusion_70_30(self) -> None:
        result = fuse_speed(50.0, 60.0)
        expected = round(0.7 * 50.0 + 0.3 * 60.0, 2)
        assert result == expected

    def test_physics_only_when_mlp_out_of_range(self) -> None:
        """MLP prediction outside [5, 100] → physics only."""
        assert fuse_speed(50.0, 200.0) == 50.0
        assert fuse_speed(50.0, 1.0) == 50.0

    def test_boundary_valid_mlp(self) -> None:
        """MLP at exact boundaries should be accepted."""
        assert fuse_speed(50.0, 5.0) == round(0.7 * 50.0 + 0.3 * 5.0, 2)
        assert fuse_speed(50.0, 100.0) == round(0.7 * 50.0 + 0.3 * 100.0, 2)


# SmoothedSpeedTracker


class TestSmoothedSpeedTracker:
    def test_none_input_returns_none(self) -> None:
        sst = SmoothedSpeedTracker()
        assert sst.update(1, None) is None

    def test_single_speed_returns_itself(self) -> None:
        sst = SmoothedSpeedTracker()
        result = sst.update(1, 50.0)
        assert result == 50.0

    def test_smoothing_averages(self) -> None:
        sst = SmoothedSpeedTracker(window_size=3)
        sst.update(1, 40.0)
        sst.update(1, 50.0)
        result = sst.update(1, 60.0)
        assert result == pytest.approx(50.0)  # mean of [40, 50, 60]

    def test_separate_tracks(self) -> None:
        sst = SmoothedSpeedTracker()
        sst.update(1, 40.0)
        sst.update(2, 80.0)
        assert sst.update(1, 40.0) == 40.0
        assert sst.update(2, 80.0) == 80.0

    def test_remove_track(self) -> None:
        sst = SmoothedSpeedTracker()
        sst.update(1, 50.0)
        sst.remove_track(1)
        assert 1 not in sst._speeds

    def test_none_clears_deque(self) -> None:
        """Updating with None should clear stale speed history for the track."""
        sst = SmoothedSpeedTracker(window_size=5)
        sst.update(1, 40.0)
        sst.update(1, 50.0)
        assert 1 in sst._speeds
        # Now send None → deque should be cleared
        result = sst.update(1, None)
        assert result is None
        assert 1 not in sst._speeds

    def test_fresh_start_after_none(self) -> None:
        """After None clears history, the next speed starts from scratch."""
        sst = SmoothedSpeedTracker(window_size=5)
        sst.update(1, 80.0)
        sst.update(1, 80.0)
        sst.update(1, None)  # clears history
        result = sst.update(1, 20.0)
        # Should be exactly 20, not averaged with old 80s
        assert result == 20.0


class TestTrackMotionStateTracker:
    def test_stationary_requires_consecutive_votes(self) -> None:
        tracker = TrackMotionStateTracker(
            enter_frames=2, exit_frames=2, exit_speed_min=6.0
        )
        history = deque([(100, 100)] * 20)
        assert tracker.update(1, history, candidate_speed=None) is False
        assert tracker.update(1, history, candidate_speed=None) is True

    def test_stationary_exit_uses_hysteresis(self) -> None:
        tracker = TrackMotionStateTracker(
            enter_frames=1, exit_frames=2, exit_speed_min=6.0
        )
        stationary_history = deque([(100, 100)] * 20)
        moving_history = deque([(100 + i * 10, 100) for i in range(20)])
        assert tracker.update(1, stationary_history, candidate_speed=None) is True
        assert tracker.update(1, moving_history, candidate_speed=12.0) is True
        assert tracker.update(1, moving_history, candidate_speed=12.0) is False

    def test_remove_track_clears_state(self) -> None:
        tracker = TrackMotionStateTracker(enter_frames=1)
        history = deque([(100, 100)] * 20)
        assert tracker.update(1, history, candidate_speed=None) is True
        tracker.remove_track(1)
        assert tracker.update(1, history, candidate_speed=None) is True


class TestRobustSpeedSummary:
    def test_empty_returns_zero(self) -> None:
        assert robust_speed_summary([]) == 0.0

    def test_single_value_is_preserved(self) -> None:
        assert robust_speed_summary([12.34]) == 12.34

    def test_outlier_is_suppressed(self) -> None:
        result = robust_speed_summary([20.0, 21.0, 19.5, 22.0, 130.0])
        assert 19.0 <= result <= 25.0


# select_model_variant


class TestSelectModelVariant:
    def test_short_clip_returns_xlarge(self) -> None:
        """Short clips (<5 min) use the heaviest model for max accuracy."""
        assert select_model_variant(30) == "yolo11x"

    def test_medium_clip_returns_large(self) -> None:
        assert select_model_variant(500) == "yolo11l"

    def test_long_clip_returns_nano(self) -> None:
        """Very long clips (>12 h) use the lightest model for speed."""
        assert select_model_variant(50000) == "yolo11n"

    def test_boundary_values(self) -> None:
        assert select_model_variant(300) == "yolo11x"  # exactly 5 min → xlarge
        assert select_model_variant(301) == "yolo11l"  # just over → large
        assert select_model_variant(1800) == "yolo11l"  # exactly 30 min → large
        assert select_model_variant(1801) == "yolo11m"  # just over → medium


# Per-vehicle-type speed limits


class TestPerTypeSpeedLimits:
    def test_bicycle_over_40_returns_none(self) -> None:
        """Bicycle at implausible speed → filtered out."""
        # Create history that would produce ~60 km/h
        history = deque([(100 + i * 15, 540) for i in range(25)])
        result = estimate_speed(
            history,
            fps=30.0,
            frame_height=1080,
            vehicle_type="bicycle",
        )
        assert result is None

    def test_car_plausible_speed(self) -> None:
        """Car within [2, 120] km/h → accepted."""
        history = deque([(100 + i * 5, 540) for i in range(25)])
        result = estimate_speed(
            history,
            fps=30.0,
            frame_height=1080,
            vehicle_type="car",
        )
        assert result is not None


# Speed estimation — noise floor, outlier clamping, rolling window


class TestSpeedNoiseFloor:
    def test_jitter_below_noise_floor_returns_none(self) -> None:
        """Sub-pixel jitter (±1px) should be zeroed → speed below SPEED_RANGE → None."""
        # 30 frames with alternating ±1px displacement (below 2px noise floor)
        history = deque([(100 + (i % 2), 540) for i in range(30)])
        result = estimate_speed(history, fps=30.0, frame_height=1080)
        assert result is None

    def test_real_movement_above_noise_floor(self) -> None:
        """Displacement of 5px/frame is well above the 2px noise floor → valid speed."""
        history = deque([(100 + i * 5, 540) for i in range(15)])
        result = estimate_speed(history, fps=30.0, frame_height=1080)
        assert result is not None
        assert result > 0


class TestSpeedOutlierClamping:
    def test_single_huge_jump_is_clamped(self) -> None:
        """One track-ID-switch jump (500px) among normal 5px moves → clamped."""
        pts = [(100 + i * 5, 540) for i in range(10)]
        # Insert a 500px jump at position 5
        pts[5] = (pts[4][0] + 500, 540)
        # Resume normal movement after the jump
        for i in range(6, 10):
            pts[i] = (pts[5][0] + (i - 5) * 5, 540)
        history = deque(pts)
        result = estimate_speed(history, fps=30.0, frame_height=1080)
        # Without clamping this would be an insane spike; with clamping
        # it either returns a plausible value or None
        if result is not None:
            assert result <= 120.0


class TestSpeedWindowFrames:
    def test_window_limits_positions_used(self) -> None:
        """With window_frames=10, only the last 10 positions matter."""
        # First 40 frames: stationary. Last 10: moving at 5px/frame.
        pts = [(100, 540)] * 40 + [(100 + i * 5, 540) for i in range(10)]
        history = deque(pts, maxlen=50)
        # Full history: mostly stationary → low speed
        # Windowed (last 10): clear movement → higher speed
        speed_windowed = estimate_speed(
            history,
            fps=30.0,
            frame_height=1080,
            window_frames=10,
        )
        speed_full = estimate_speed(
            history,
            fps=30.0,
            frame_height=1080,
            window_frames=50,
        )
        # The windowed estimate should be higher since it ignores the
        # 40 stationary frames at the start.
        if speed_windowed is not None and speed_full is not None:
            assert speed_windowed > speed_full


class TestNearZeroMotion:
    def test_relaxed_stationary_band_detects_near_zero_motion(self) -> None:
        history = deque([(100, 100), (101, 100), (102, 101), (103, 101)])
        assert is_near_zero_motion(history) is True

    def test_large_motion_is_not_near_zero(self) -> None:
        history = deque([(100, 100), (120, 100), (140, 100), (160, 100)])
        assert is_near_zero_motion(history) is False


