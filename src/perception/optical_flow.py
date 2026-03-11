"""Optical-flow camera-motion estimation for the VAAET production pipeline.

Uses Lucas-Kanade sparse optical flow on a regular feature-point grid to
estimate global camera motion, then applies a running mean for temporal
smoothing.  This compensates for pan/tilt/zoom of the SISE cameras on the
General Manuel Belgrano bridge.

References:
    - Legacy implementation: ``archive/00_bootstrap/01_legacy_collection.ipynb``
      (``VAAETHybrid.calculate_global_motion()``)
    - ADR-009 §Perception
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from src.config import (
    OPTICAL_FLOW_GRID_STEP,
    OPTICAL_FLOW_MAX_LEVEL,
    OPTICAL_FLOW_RUNNING_MEAN,
    OPTICAL_FLOW_WIN_SIZE,
)

__all__ = ["OpticalFlowEstimator"]


class OpticalFlowEstimator:
    """Estimate global camera motion using Lucas-Kanade sparse optical flow.

    The estimator places feature points on a regular grid (every
    ``grid_step`` pixels), computes sparse flow between consecutive grey
    frames, takes the **median** displacement as the raw global motion,
    then applies a running mean over the last ``running_mean_window``
    frames for temporal stability.

    Args:
        grid_step: Pixel spacing between feature points.
        win_size: Lucas-Kanade window size ``(w, h)``.
        max_level: Number of pyramid levels for LK tracking.
        running_mean_window: Number of frames for motion smoothing.
    """

    def __init__(
        self,
        grid_step: int = OPTICAL_FLOW_GRID_STEP,
        win_size: tuple[int, int] = OPTICAL_FLOW_WIN_SIZE,
        max_level: int = OPTICAL_FLOW_MAX_LEVEL,
        running_mean_window: int = OPTICAL_FLOW_RUNNING_MEAN,
    ) -> None:
        self.grid_step = grid_step
        self.win_size = win_size
        self.max_level = max_level

        self._prev_gray: np.ndarray | None = None
        self._motion_history: deque[np.ndarray] = deque(
            maxlen=running_mean_window,
        )
        # LK parameters dict (passed directly to cv2.calcOpticalFlowPyrLK)
        self._lk_params: dict[str, Any] = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

    # ── Public API ─────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Compute the smoothed global motion vector for *frame*.

        On the first call the estimator stores the greyscale image and
        returns a zero vector.  On subsequent calls it returns the
        running-mean-smoothed global motion.

        Args:
            frame: BGR frame (OpenCV convention).

        Returns:
            2-D float array ``[dx, dy]`` representing the smoothed global
            camera motion in pixels.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros(2, dtype=float)

        raw_motion = self._compute_raw_motion(gray)
        self._motion_history.append(raw_motion)
        self._prev_gray = gray

        # Running mean across recent frames
        return np.mean(self._motion_history, axis=0)

    def reset(self) -> None:
        """Clear internal state (call between clips)."""
        self._prev_gray = None
        self._motion_history.clear()

    # ── Private helpers ────────────────────────────────────────────────

    def _build_grid_points(self, h: int, w: int) -> np.ndarray:
        """Generate feature points on a regular grid.

        Args:
            h: Frame height.
            w: Frame width.

        Returns:
            Array of shape ``(N, 1, 2)`` with ``float32`` grid coordinates.
        """
        ys = np.arange(0, h, self.grid_step)
        xs = np.arange(0, w, self.grid_step)
        grid = np.array(
            np.meshgrid(xs, ys),
            dtype=np.float32,
        ).T.reshape(-1, 1, 2)
        return grid

    def _compute_raw_motion(self, gray: np.ndarray) -> np.ndarray:
        """Run LK optical flow and return the raw median displacement.

        Args:
            gray: Current greyscale frame.

        Returns:
            2-D float array ``[dx, dy]``.
        """
        h, w = gray.shape[:2]
        pts = self._build_grid_points(h, w)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            pts,
            None,
            **self._lk_params,
        )

        if new_pts is None or status is None:
            return np.zeros(2, dtype=float)

        # Keep only points that were successfully tracked
        good_mask = status.ravel() == 1
        if not np.any(good_mask):
            return np.zeros(2, dtype=float)

        old_good = pts[good_mask].reshape(-1, 2)
        new_good = new_pts[good_mask].reshape(-1, 2)
        displacements = new_good - old_good

        # Median is robust against outliers (moving vehicles)
        return np.median(displacements, axis=0)
