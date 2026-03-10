"""Speed estimation for the VAAET production pipeline.

Provides physics-based speed calculation from tracked vehicle centroids,
with optional perspective correction and camera-motion compensation.

NOTE: This is a skeleton — full implementation is tracked for Module 2
(production notebook).  See ADR-009.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.config import SPEED_RANGE

__all__ = [
    "estimate_speed",
    "compensate_camera_motion",
    "get_perspective_factor",
]


def get_perspective_factor(
    y: float,
    frame_height: int,
    near_factor: float = 1.8,
    mid_factor: float = 1.0,
    far_factor: float = 0.6,
) -> float:
    """Return a perspective correction factor based on vertical position.

    Objects near the bottom of the frame (close to camera) appear to move
    faster in pixel space; objects near the top (far) appear slower.

    Args:
        y: Vertical coordinate of the object centroid.
        frame_height: Height of the video frame in pixels.
        near_factor: Scale factor for objects near the camera.
        mid_factor: Scale factor for the middle zone.
        far_factor: Scale factor for distant objects.

    Returns:
        A multiplicative correction factor.
    """
    ratio = y / max(frame_height, 1)
    if ratio > 0.66:
        return near_factor
    elif ratio > 0.33:
        return mid_factor
    return far_factor


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
    pixels_per_meter: float = 12.0,
    frame_height: int = 1080,
    global_motion: np.ndarray | None = None,
) -> float | None:
    """Estimate vehicle speed in km/h from its centroid history.

    The calculation follows the physics-based approach:
      1. Compute cumulative Euclidean displacement in pixel space.
      2. Optionally subtract global camera motion (optical flow).
      3. Apply perspective correction based on vertical position.
      4. Convert pixels → meters → km/h.
      5. Filter by plausibility range.

    Args:
        history: Deque of ``(cx, cy)`` centroid positions.
        fps: Video frames per second.
        pixels_per_meter: Calibration factor.
        frame_height: Frame height for perspective correction.
        global_motion: Optional global motion vector to compensate.

    Returns:
        Estimated speed in km/h, or *None* if implausible / insufficient data.
    """
    if len(history) < 2:
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

    # Plausibility filter
    if not (SPEED_RANGE[0] <= speed_kmh <= SPEED_RANGE[1]):
        return None

    return round(speed_kmh, 2)
