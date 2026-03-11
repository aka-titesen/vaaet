"""Speed estimation for the VAAET production pipeline.

Provides physics-based speed calculation from tracked vehicle centroids,
with perspective correction, camera-motion compensation, stationary
detection, and optional MLP 70/30 fusion smoothing.

References:
    - Legacy: ``VAAETHybrid.calculate_enhanced_speed()``,
      ``VAAETHybrid.is_stationary()`` in ``archive/00_bootstrap/``
    - ADR-009 §Perception, ADR-004
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.config import (
    PERSPECTIVE_ZONES,
    PIXELS_PER_METER,
    SPEED_LIMITS_PER_TYPE,
    SPEED_MIN_TRACK_LENGTH,
    SPEED_MLP_VALID_RANGE,
    SPEED_MLP_WEIGHT,
    SPEED_PHYSICS_WEIGHT,
    SPEED_RANGE,
    STATIONARY_AVG_FRAME_MAX,
    STATIONARY_MAX_FRAME_MAX,
    STATIONARY_MAX_SEGMENT_MAX,
    STATIONARY_STD_MAX,
    STATIONARY_TOTAL_DISP_MAX,
)

__all__ = [
    "estimate_speed",
    "compensate_camera_motion",
    "get_perspective_factor",
    "is_stationary",
    "fuse_speed",
    "SmoothedSpeedTracker",
]


def get_perspective_factor(
    y: float,
    frame_height: int,
    near_factor: float | None = None,
    mid_factor: float | None = None,
    far_factor: float | None = None,
) -> float:
    """Return a perspective correction factor based on vertical position.

    Objects near the bottom of the frame (close to camera) appear to move
    faster in pixel space; objects near the top (far) appear slower.
    Factors default to the values in ``config.PERSPECTIVE_ZONES``.

    Args:
        y: Vertical coordinate of the object centroid.
        frame_height: Height of the video frame in pixels.
        near_factor: Override scale factor for near zone.
        mid_factor: Override scale factor for mid zone.
        far_factor: Override scale factor for far zone.

    Returns:
        A multiplicative correction factor.
    """
    _near = (
        near_factor if near_factor is not None else PERSPECTIVE_ZONES["near"]["factor"]
    )
    _mid = mid_factor if mid_factor is not None else PERSPECTIVE_ZONES["mid"]["factor"]
    _far = far_factor if far_factor is not None else PERSPECTIVE_ZONES["far"]["factor"]

    ratio = y / max(frame_height, 1)
    if ratio > PERSPECTIVE_ZONES["near"]["threshold"]:
        return _near
    elif ratio > PERSPECTIVE_ZONES["mid"]["threshold"]:
        return _mid
    return _far


def compensate_camera_motion(
    displacement: np.ndarray,
    global_motion: np.ndarray,
) -> np.ndarray:
    """Subtract estimated camera motion from a vehicle displacement vector.

    Args:
        displacement: 2-D displacement vector of the tracked vehicle.
        global_motion: Estimated global motion vector (median optical flow).

    Returns:
        Compensated displacement vector.
    """
    return displacement - global_motion


def estimate_speed(
    history: deque,
    fps: float,
    pixels_per_meter: float = PIXELS_PER_METER,
    frame_height: int = 1080,
    global_motion: np.ndarray | None = None,
    vehicle_type: str | None = None,
    min_track_length: int = SPEED_MIN_TRACK_LENGTH,
) -> float | None:
    """Estimate vehicle speed in km/h from its centroid history.

    The calculation follows the physics-based approach:
      1. Compute cumulative Euclidean displacement in pixel space.
      2. Optionally subtract global camera motion (optical flow).
      3. Apply perspective correction based on vertical position.
      4. Convert pixels → meters → km/h.
      5. Filter by plausibility range (global and per-type).

    Args:
        history: Deque of ``(cx, cy)`` centroid positions.
        fps: Video frames per second.
        pixels_per_meter: Calibration factor.
        frame_height: Frame height for perspective correction.
        global_motion: Optional global motion vector to compensate.
        vehicle_type: Optional vehicle type for per-type speed limits.
        min_track_length: Minimum frames in history for reliable estimate.

    Returns:
        Estimated speed in km/h, or *None* if implausible / insufficient data.
    """
    if len(history) < max(2, min_track_length):
        return None

    positions = np.array(history, dtype=float)
    displacements = np.diff(positions, axis=0)

    # Camera-motion compensation
    if global_motion is not None:
        displacements = displacements - global_motion

    # Total displacement in pixels
    total_px = float(np.sum(np.linalg.norm(displacements, axis=1)))

    # Perspective correction (use average Y)
    avg_y = float(positions[:, 1].mean())
    pf = get_perspective_factor(avg_y, frame_height)
    total_px *= pf

    # Convert to meters
    distance_m = total_px / pixels_per_meter

    # Time span
    n_frames = len(history) - 1
    time_s = n_frames / max(fps, 1)

    if time_s <= 0:
        return None

    speed_kmh = (distance_m / time_s) * 3.6

    # Per-vehicle-type plausibility filter
    if vehicle_type and vehicle_type in SPEED_LIMITS_PER_TYPE:
        lo, hi = SPEED_LIMITS_PER_TYPE[vehicle_type]
        if not (lo <= speed_kmh <= hi):
            return None
    else:
        # Fallback to global range
        if not (SPEED_RANGE[0] <= speed_kmh <= SPEED_RANGE[1]):
            return None

    return round(speed_kmh, 2)


# ── Stationary detection ──────────────────────────────────────────────


def is_stationary(history: deque) -> bool:
    """Determine whether a tracked vehicle is stationary.

    Uses the **AND-conjunction** of five criteria (all must be true):
      1. Total displacement < ``STATIONARY_TOTAL_DISP_MAX``
      2. Max single-segment displacement < ``STATIONARY_MAX_SEGMENT_MAX``
      3. Displacement std-dev < ``STATIONARY_STD_MAX``
      4. Average per-frame displacement < ``STATIONARY_AVG_FRAME_MAX``
      5. Max per-frame displacement < ``STATIONARY_MAX_FRAME_MAX``

    **Do not relax to OR without an ADR** (see AGENTS.md).

    Args:
        history: Deque of ``(cx, cy)`` centroid positions.

    Returns:
        ``True`` if the vehicle is considered stationary.
    """
    if len(history) < 2:
        return True

    positions = np.array(history, dtype=float)
    frame_disps = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    total_disp = float(np.sum(frame_disps))
    max_segment = float(np.max(frame_disps))
    std_disp = float(np.std(frame_disps))
    avg_frame = float(np.mean(frame_disps))
    max_frame = float(np.max(frame_disps))

    return (
        total_disp < STATIONARY_TOTAL_DISP_MAX
        and max_segment < STATIONARY_MAX_SEGMENT_MAX
        and std_disp < STATIONARY_STD_MAX
        and avg_frame < STATIONARY_AVG_FRAME_MAX
        and max_frame < STATIONARY_MAX_FRAME_MAX
    )


# ── MLP 70/30 speed fusion ───────────────────────────────────────────


def fuse_speed(
    physics_speed: float,
    mlp_speed: float | None,
) -> float:
    """Fuse physics-based and MLP-predicted speed estimates.

    The formula is ``0.7 * physics + 0.3 * mlp`` when the MLP prediction
    falls within a plausible range, otherwise uses physics-only.

    **Do not alter the 70/30 split without experimental evidence**
    (see AGENTS.md).

    Args:
        physics_speed: Speed from the physics-based estimator (km/h).
        mlp_speed: Speed from the MLP smoother (km/h), or *None*.

    Returns:
        Fused speed estimate in km/h.
    """
    if mlp_speed is None:
        return physics_speed

    lo, hi = SPEED_MLP_VALID_RANGE
    if lo <= mlp_speed <= hi:
        return round(
            SPEED_PHYSICS_WEIGHT * physics_speed + SPEED_MLP_WEIGHT * mlp_speed,
            2,
        )
    return physics_speed


# ── SmoothedSpeedTracker ──────────────────────────────────────────────


class SmoothedSpeedTracker:
    """Per-track speed smoothing with optional MLP fusion.

    Maintains a per-vehicle running window of raw speed estimates and
    optionally fuses with an MLP regressor prediction.

    Args:
        window_size: Number of recent speed estimates to average.
        mlp_model: Optional scikit-learn ``MLPRegressor`` instance.
            Expected to accept a 10-feature vector and return a single
            speed value.
    """

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
        """Record a new physics speed estimate and return the smoothed value.

        Args:
            track_id: Unique track identifier.
            physics_speed: Raw physics-based speed (km/h), or *None*.
            mlp_features: Optional 10-element feature vector for the MLP.

        Returns:
            Smoothed (and optionally fused) speed, or *None*.
        """
        if physics_speed is None:
            return None

        if track_id not in self._speeds:
            self._speeds[track_id] = deque(maxlen=self.window_size)

        self._speeds[track_id].append(physics_speed)
        avg_physics = float(np.mean(self._speeds[track_id]))

        # MLP fusion
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
