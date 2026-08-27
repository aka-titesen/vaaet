# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Speed estimation for the VAAET production pipeline.

Provides physics-based speed calculation from tracked vehicle centroids,
with perspective correction, camera-motion compensation, stationary
detection, and optional MLP 70/30 fusion smoothing.

See ADR-0004, ADR-0006 and ADR-0009 for the decision context.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from vaaet.settings import (
    NEAR_ZERO_AVG_FRAME_MAX,
    NEAR_ZERO_MAX_FRAME_MAX,
    NEAR_ZERO_MAX_SEGMENT_MAX,
    NEAR_ZERO_STD_MAX,
    NEAR_ZERO_TOTAL_DISP_MAX,
    OPTICAL_FLOW_MIN_TRACKING_RATIO,
    PERSPECTIVE_BLEND_BAND,
    PERSPECTIVE_ZONES,
    PIXELS_PER_METER,
    SPEED_DISPLACEMENT_NOISE_FLOOR,
    SPEED_ESTIMATION_WINDOW,
    SPEED_LIMITS_PER_TYPE,
    SPEED_MIN_TRACK_LENGTH,
    SPEED_MLP_VALID_RANGE,
    SPEED_MLP_WEIGHT,
    SPEED_PHYSICS_WEIGHT,
    SPEED_RANGE,
    SPEED_RECOVERY_SKIP_GAP,
    SPEED_ROBUST_OUTLIER_SIGMA,
    SPEED_ROBUST_TRIM_RATIO,
    STATIONARY_AVG_FRAME_MAX,
    STATIONARY_ENTRY_FRAMES,
    STATIONARY_EXIT_FRAMES,
    STATIONARY_EXIT_SPEED_MIN,
    STATIONARY_MAX_FRAME_MAX,
    STATIONARY_MAX_SEGMENT_MAX,
    STATIONARY_STD_MAX,
    STATIONARY_TOTAL_DISP_MAX,
)

__all__ = [
    "estimate_speed",
    "compensate_camera_motion",
    "get_perspective_factor",
    "is_near_zero_motion",
    "is_speed_measurement_reliable",
    "is_stationary",
    "robust_speed_summary",
    "fuse_speed",
    "SmoothedSpeedTracker",
    "TrackMotionStateTracker",
]


def _lerp(start: float, end: float, alpha: float) -> float:
    """Linearly interpolate between two values."""
    return start + (end - start) * alpha


def _recent_displacement_norms(history: deque, window: int = 8) -> np.ndarray:
    """Return recent per-frame displacement magnitudes for a track history."""
    if len(history) < 2:
        return np.array([], dtype=float)
    positions = np.array(list(history)[-window:], dtype=float)
    displacements = np.diff(positions, axis=0)
    return np.linalg.norm(displacements, axis=1)


def _motion_stats(history: deque) -> tuple[float, float, float, float, float]:
    if len(history) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    positions = np.array(history, dtype=float)
    frame_disps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if frame_disps.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    total_disp = float(np.sum(frame_disps))
    max_segment = float(np.max(frame_disps))
    std_disp = float(np.std(frame_disps))
    avg_frame = float(np.mean(frame_disps))
    max_frame = float(np.max(frame_disps))
    return total_disp, max_segment, std_disp, avg_frame, max_frame


def get_perspective_factor(
    y: float,
    frame_height: int,
    near_factor: float | None = None,
    mid_factor: float | None = None,
    far_factor: float | None = None,
) -> float:
    """Return a perspective correction factor based on vertical position."""
    _near = near_factor if near_factor is not None else PERSPECTIVE_ZONES["near"]["factor"]
    _mid = mid_factor if mid_factor is not None else PERSPECTIVE_ZONES["mid"]["factor"]
    _far = far_factor if far_factor is not None else PERSPECTIVE_ZONES["far"]["factor"]

    ratio = y / max(frame_height, 1)
    near_threshold = PERSPECTIVE_ZONES["near"]["threshold"]
    mid_threshold = PERSPECTIVE_ZONES["mid"]["threshold"]
    blend_band = max(PERSPECTIVE_BLEND_BAND, 0.0)

    if blend_band <= 0:
        if ratio > near_threshold:
            return _near
        if ratio > mid_threshold:
            return _mid
        return _far

    if ratio >= near_threshold + blend_band:
        return _near
    if ratio <= mid_threshold - blend_band:
        return _far

    if near_threshold - blend_band <= ratio <= near_threshold + blend_band:
        alpha = (ratio - (near_threshold - blend_band)) / (2.0 * blend_band)
        return _lerp(_mid, _near, float(np.clip(alpha, 0.0, 1.0)))

    if mid_threshold - blend_band <= ratio <= mid_threshold + blend_band:
        alpha = (ratio - (mid_threshold - blend_band)) / (2.0 * blend_band)
        return _lerp(_far, _mid, float(np.clip(alpha, 0.0, 1.0)))

    if ratio > mid_threshold:
        return _mid
    return _far


def compensate_camera_motion(
    displacement: np.ndarray,
    global_motion: np.ndarray,
) -> np.ndarray:
    """Subtract estimated camera motion from a vehicle displacement vector."""
    return displacement - global_motion


def estimate_speed(
    history: deque,
    fps: float,
    pixels_per_meter: float = PIXELS_PER_METER,
    frame_height: int = 1080,
    global_motion: np.ndarray | None = None,
    vehicle_type: str | None = None,
    min_track_length: int = SPEED_MIN_TRACK_LENGTH,
    window_frames: int = SPEED_ESTIMATION_WINDOW,
    noise_floor: float = SPEED_DISPLACEMENT_NOISE_FLOOR,
) -> float | None:
    """Estimate vehicle speed in km/h from its centroid history."""
    if len(history) < max(2, min_track_length):
        return None

    window = min(window_frames, len(history))
    positions = np.array(list(history)[-window:], dtype=float)
    displacements = np.diff(positions, axis=0)

    if global_motion is not None:
        displacements = displacements - global_motion

    norms = np.linalg.norm(displacements, axis=1)
    norms[norms < noise_floor] = 0.0

    if len(norms) >= 3:
        med = float(np.median(norms[norms > 0])) if np.any(norms > 0) else 0.0
        if med > 0:
            norms = np.minimum(norms, 3.0 * med)

    total_px = float(np.sum(norms))
    avg_y = float(positions[:, 1].mean())
    pf = get_perspective_factor(avg_y, frame_height)
    total_px *= pf

    distance_m = total_px / pixels_per_meter
    n_frames = len(positions) - 1
    time_s = n_frames / max(fps, 1)
    if time_s <= 0:
        return None

    speed_kmh = (distance_m / time_s) * 3.6

    if vehicle_type and vehicle_type in SPEED_LIMITS_PER_TYPE:
        lo, hi = SPEED_LIMITS_PER_TYPE[vehicle_type]
        if not (lo <= speed_kmh <= hi):
            return None
    elif not (SPEED_RANGE[0] <= speed_kmh <= SPEED_RANGE[1]):
        return None

    return round(speed_kmh, 2)


def is_near_zero_motion(history: deque) -> bool:
    """Return ``True`` when motion is minimal but not necessarily stationary."""
    total_disp, max_segment, std_disp, avg_frame, max_frame = _motion_stats(history)
    return (
        total_disp < NEAR_ZERO_TOTAL_DISP_MAX
        and max_segment < NEAR_ZERO_MAX_SEGMENT_MAX
        and std_disp < NEAR_ZERO_STD_MAX
        and avg_frame < NEAR_ZERO_AVG_FRAME_MAX
        and max_frame < NEAR_ZERO_MAX_FRAME_MAX
    )


# Stationary detection


def is_stationary(history: deque) -> bool:
    """Determine whether a tracked vehicle is stationary."""
    total_disp, max_segment, std_disp, avg_frame, max_frame = _motion_stats(history)
    return (
        total_disp < STATIONARY_TOTAL_DISP_MAX
        and max_segment < STATIONARY_MAX_SEGMENT_MAX
        and std_disp < STATIONARY_STD_MAX
        and avg_frame < STATIONARY_AVG_FRAME_MAX
        and max_frame < STATIONARY_MAX_FRAME_MAX
    )


def is_speed_measurement_reliable(
    history: deque,
    flow_tracking_ratio: float = 1.0,
    recovered_after_gap: int = 0,
    min_flow_tracking_ratio: float = OPTICAL_FLOW_MIN_TRACKING_RATIO,
    recovery_skip_gap: int = SPEED_RECOVERY_SKIP_GAP,
) -> bool:
    """Return whether a track's speed estimate is reliable enough to use."""
    if len(history) < max(2, SPEED_MIN_TRACK_LENGTH):
        return False
    if recovered_after_gap >= recovery_skip_gap:
        return False
    if flow_tracking_ratio < min_flow_tracking_ratio:
        return False

    norms = _recent_displacement_norms(history)
    if len(norms) < 3:
        return True

    non_zero = norms[norms > 0]
    if len(non_zero) == 0:
        return True

    median_baseline = float(np.median(non_zero))
    if median_baseline <= 0:
        return True

    anomaly_limit = max(4.0 * median_baseline, SPEED_DISPLACEMENT_NOISE_FLOOR * 3.0)
    return float(np.max(norms)) <= anomaly_limit


def robust_speed_summary(
    speeds: list[float] | np.ndarray,
    trim_ratio: float = SPEED_ROBUST_TRIM_RATIO,
    outlier_sigma: float = SPEED_ROBUST_OUTLIER_SIGMA,
) -> float:
    """Aggregate speeds using a robust mean that suppresses isolated spikes."""
    arr = np.asarray(speeds, dtype=float)
    if arr.size == 0:
        return 0.0

    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0.0]
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return round(float(arr[0]), 2)

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 0:
        modified_z = 0.6745 * (arr - median) / mad
        filtered = arr[np.abs(modified_z) <= outlier_sigma]
        if filtered.size > 0:
            arr = filtered

    if arr.size >= 4 and 0.0 < trim_ratio < 0.5:
        trim_n = int(arr.size * trim_ratio)
        if trim_n > 0 and (arr.size - (2 * trim_n)) >= 1:
            arr = np.sort(arr)[trim_n:-trim_n]

    if arr.size == 0:
        return 0.0
    return round(float(np.mean(arr)), 2)


# MLP 70/30 speed fusion


def fuse_speed(
    physics_speed: float,
    mlp_speed: float | None,
) -> float:
    """Fuse physics-based and MLP-predicted speed estimates."""
    if mlp_speed is None:
        return physics_speed

    lo, hi = SPEED_MLP_VALID_RANGE
    if lo <= mlp_speed <= hi:
        return round(
            SPEED_PHYSICS_WEIGHT * physics_speed + SPEED_MLP_WEIGHT * mlp_speed,
            2,
        )
    return physics_speed


class SmoothedSpeedTracker:
    """Per-track speed smoothing with optional MLP fusion."""

    def __init__(
        self,
        window_size: int = 10,
        mlp_model: Any = None,
    ) -> None:
        self.window_size = window_size
        self.mlp_model = mlp_model
        self._speeds: dict[int, deque] = {}

    def update(
        self,
        track_id: int,
        physics_speed: float | None,
        mlp_features: np.ndarray | None = None,
    ) -> float | None:
        """Record a new physics speed estimate and return the smoothed value."""
        if physics_speed is None:
            self._speeds.pop(track_id, None)
            return None

        if track_id not in self._speeds:
            self._speeds[track_id] = deque(maxlen=self.window_size)

        self._speeds[track_id].append(physics_speed)
        avg_physics = float(np.mean(self._speeds[track_id]))

        mlp_speed: float | None = None
        if self.mlp_model is not None and mlp_features is not None:
            try:
                pred = self.mlp_model.predict(mlp_features.reshape(1, -1))
                mlp_speed = float(pred[0])
            except Exception:
                mlp_speed = None

        return fuse_speed(avg_physics, mlp_speed)

    def remove_track(self, track_id: int) -> None:
        """Remove speed history for a track that has been pruned."""
        self._speeds.pop(track_id, None)


class TrackMotionStateTracker:
    """Hysteresis-based stationary state tracker per vehicle track."""

    def __init__(
        self,
        enter_frames: int = STATIONARY_ENTRY_FRAMES,
        exit_frames: int = STATIONARY_EXIT_FRAMES,
        exit_speed_min: float = STATIONARY_EXIT_SPEED_MIN,
    ) -> None:
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.exit_speed_min = exit_speed_min
        self._stationary: dict[int, bool] = {}
        self._enter_votes: dict[int, int] = {}
        self._exit_votes: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        history: deque,
        candidate_speed: float | None = None,
    ) -> bool:
        """Update and return the hysteresis-filtered stationary state."""
        raw_stationary = is_stationary(history)
        current = self._stationary.get(track_id, False)

        if raw_stationary:
            self._enter_votes[track_id] = self._enter_votes.get(track_id, 0) + 1
            self._exit_votes[track_id] = 0
        else:
            self._enter_votes[track_id] = 0
            moving_vote = candidate_speed is not None and candidate_speed >= self.exit_speed_min
            self._exit_votes[track_id] = self._exit_votes.get(track_id, 0) + (
                1 if moving_vote else 0
            )

        if current:
            if self._exit_votes.get(track_id, 0) >= self.exit_frames:
                current = False
        elif self._enter_votes.get(track_id, 0) >= self.enter_frames:
            current = True

        self._stationary[track_id] = current
        return current

    def remove_track(self, track_id: int) -> None:
        """Forget all state for a track that disappeared or was pruned."""
        self._stationary.pop(track_id, None)
        self._enter_votes.pop(track_id, None)
        self._exit_votes.pop(track_id, None)
