# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Tracker SORT liviano para el pipeline operativo de VAAET.

Implementa una variante simplificada de SORT basada en la distancia euclídea
entre centroides. Consultar ADR-0003 y ADR-0009 para el contexto de la decisión.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vaaet.settings import (
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
    """Vehículo seguido con historial de posiciones y conteo único."""

    track_id: int
    vehicle_type: str
    centroid: tuple[int, int]
    history: deque = field(
        default_factory=lambda: deque(maxlen=TRACKER_HISTORY_MAXLEN),
    )
    frames_since_seen: int = 0
    recovered_after_gap: int = 0
    total_frames: int = 0
    counted: bool = False  # Se activa después del único conteo permitido.

    def update(self, centroid: tuple[int, int]) -> None:
        """Actualiza el track con la posición de su nuevo centroide."""
        gap = self.frames_since_seen
        self.centroid = centroid
        self.history.append(centroid)
        self.frames_since_seen = 0
        self.recovered_after_gap = gap if gap > 0 else 0
        self.total_frames += 1

    def mark_counted(self) -> bool:
        """Marca el track como contado y sólo devuelve ``True`` la primera vez."""
        if self.counted:
            return False
        self.counted = True
        return True


class SORTTracker:
    """Tracker SORT por distancia euclídea para centroides de vehículos.

    Args:
        max_distance: Distancia máxima en píxeles para asociar una detección.
        max_lost: Cantidad de frames ausentes antes de eliminar un track.
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
        self.last_pruned_track_ids: list[int] = []

    @property
    def active_tracks(self) -> list[Track]:
        """Devuelve los tracks activos, es decir, no ausentes en este frame."""
        return [t for t in self._tracks if t.frames_since_seen == 0]

    @property
    def all_tracks(self) -> list[Track]:
        """Devuelve todos los tracks, incluso los ausentes aún no depurados."""
        return list(self._tracks)

    def reset(self) -> None:
        """Restablece el estado entre clips o segmentos de análisis."""
        self._tracks.clear()
        self._next_id = 1
        self.last_pruned_track_ids = []

    def update(
        self,
        detections: list[tuple[tuple[int, int], str]],
    ) -> list[Track]:
        """Asocia detecciones nuevas y conserva la identidad ordenada de cada track."""
        matched_tracks, matched_detections = self._match_existing_tracks(detections)
        self._mark_unmatched_tracks(matched_tracks)
        self._append_unmatched_detections(detections, matched_detections)
        self._prune()
        return self.active_tracks

    def _match_existing_tracks(
        self, detections: list[tuple[tuple[int, int], str]]
    ) -> tuple[set[int], set[int]]:
        if not detections or not self._tracks:
            return set(), set()
        distances = self._distance_matrix(detections)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for flat_index in np.argsort(distances, axis=None):
            track_index = int(flat_index // distances.shape[1])
            detection_index = int(flat_index % distances.shape[1])
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            if distances[track_index, detection_index] > self.max_distance:
                break
            if self._tracks[track_index].vehicle_type != detections[detection_index][1]:
                continue
            self._tracks[track_index].update(detections[detection_index][0])
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
        return matched_tracks, matched_detections

    def _distance_matrix(self, detections: list[tuple[tuple[int, int], str]]) -> np.ndarray:
        track_centroids = np.array([track.centroid for track in self._tracks], dtype=float)
        detection_centroids = np.array([centroid for centroid, _ in detections], dtype=float)
        differences = track_centroids[:, None, :] - detection_centroids[None, :, :]
        return np.linalg.norm(differences, axis=2)

    def _mark_unmatched_tracks(self, matched_tracks: set[int]) -> None:
        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.frames_since_seen += 1

    def _append_unmatched_detections(
        self,
        detections: list[tuple[tuple[int, int], str]],
        matched_detections: set[int],
    ) -> None:
        for index, (centroid, vehicle_type) in enumerate(detections):
            if index in matched_detections:
                continue
            track = Track(self._next_id, vehicle_type, centroid)
            track.history.append(centroid)
            track.total_frames = 1
            self._tracks.append(track)
            self._next_id += 1

    def _prune(self) -> None:
        """Elimina tracks que superaron el máximo de frames ausentes."""
        pruned = [t.track_id for t in self._tracks if t.frames_since_seen > self.max_lost]
        self.last_pruned_track_ids = pruned
        self._tracks = [t for t in self._tracks if t.frames_since_seen <= self.max_lost]
