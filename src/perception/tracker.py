"""Lightweight SORT tracker for the VAAET production pipeline.

Implements a simplified SORT (Simple Online and Realtime Tracking) algorithm
based on Euclidean distance matching between centroids.  This mirrors the
tracking approach used in the archived bootstrap module but is extracted into
a clean, testable class.

References:
    - Legacy: ``VAAETHybrid._find_or_create_track()`` and
      ``VAAETHybrid.update_tracking()`` in ``archive/00_bootstrap/``
    - ADR-003, ADR-009 §Perception
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import (
    TRACKER_HISTORY_MAXLEN,
    TRACKER_MAX_DISTANCE,
    TRACKER_MAX_LOST,
)

__all__ = [
    "Track",
    "SORTTracker",
]


@dataclass
class Track:
    """A tracked vehicle with position history and a count-once flag."""

    track_id: int
    vehicle_type: str
    centroid: tuple[int, int]
    history: deque = field(
        default_factory=lambda: deque(maxlen=TRACKER_HISTORY_MAXLEN),
    )
    frames_since_seen: int = 0
    total_frames: int = 0
    counted: bool = False  # True after the vehicle has been tallied once

    def update(self, centroid: tuple[int, int]) -> None:
        """Update the track with a new centroid position."""
        self.centroid = centroid
        self.history.append(centroid)
        self.frames_since_seen = 0
        self.total_frames += 1

    def mark_counted(self) -> bool:
        """Mark this track as counted. Returns True on first call only."""
        if self.counted:
            return False
        self.counted = True
        return True


class SORTTracker:
    """Euclidean-distance SORT tracker for vehicle centroids.

    Args:
        max_distance: Maximum pixel distance for matching.
        max_lost: Frames before a track is removed.
    """

    def __init__(
        self,
        max_distance: float = TRACKER_MAX_DISTANCE,
        max_lost: int = TRACKER_MAX_LOST,
    ) -> None:
        self.max_distance = max_distance
        self.max_lost = max_lost
        self._tracks: list[Track] = []
        self._next_id: int = 1

    @property
    def active_tracks(self) -> list[Track]:
        """Return currently active (non-lost) tracks."""
        return [t for t in self._tracks if t.frames_since_seen == 0]

    @property
    def all_tracks(self) -> list[Track]:
        """Return all tracks (including lost but not yet pruned)."""
        return list(self._tracks)

    def reset(self) -> None:
        """Clear all tracks (call between clips or minutes)."""
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: list[tuple[tuple[int, int], str]],
    ) -> list[Track]:
        """Match new detections to existing tracks.

        Args:
            detections: List of ``(centroid, vehicle_type)`` tuples.

        Returns:
            List of all active tracks after the update.
        """
        if not detections:
            for track in self._tracks:
                track.frames_since_seen += 1
            self._prune()
            return self.active_tracks

        # Build cost matrix (Euclidean distance)
        det_centroids = np.array([d[0] for d in detections], dtype=float)
        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        if self._tracks:
            trk_centroids = np.array(
                [t.centroid for t in self._tracks],
                dtype=float,
            )
            # Pairwise distance matrix: (n_tracks, n_detections)
            diff = trk_centroids[:, None, :] - det_centroids[None, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # Greedy matching (lowest distance first)
            flat_order = np.argsort(dists, axis=None)
            for flat_idx in flat_order:
                t_idx = int(flat_idx // dists.shape[1])
                d_idx = int(flat_idx % dists.shape[1])
                if t_idx in matched_trk or d_idx in matched_det:
                    continue
                if dists[t_idx, d_idx] > self.max_distance:
                    break
                # Match only same vehicle type
                if self._tracks[t_idx].vehicle_type != detections[d_idx][1]:
                    continue
                self._tracks[t_idx].update(detections[d_idx][0])
                matched_trk.add(t_idx)
                matched_det.add(d_idx)

        # Increment lost counter for unmatched tracks
        for i, track in enumerate(self._tracks):
            if i not in matched_trk:
                track.frames_since_seen += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det:
                new_track = Track(
                    track_id=self._next_id,
                    vehicle_type=det[1],
                    centroid=det[0],
                )
                new_track.history.append(det[0])
                new_track.total_frames = 1
                self._tracks.append(new_track)
                self._next_id += 1

        self._prune()
        return self.active_tracks

    def _prune(self) -> None:
        """Remove tracks that have been lost for too long."""
        self._tracks = [t for t in self._tracks if t.frames_since_seen <= self.max_lost]
