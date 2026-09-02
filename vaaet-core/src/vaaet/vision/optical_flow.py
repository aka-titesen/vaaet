# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Estimación del movimiento de cámara mediante flujo óptico.

Aplica Lucas-Kanade sobre una grilla regular, toma la mediana del desplazamiento
y suaviza temporalmente el resultado para compensar movimientos de cámara.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from vaaet.settings import (
    OPTICAL_FLOW_BORDER_MARGIN,
    OPTICAL_FLOW_GRID_STEP,
    OPTICAL_FLOW_MAX_LEVEL,
    OPTICAL_FLOW_MIN_TRACKING_RATIO,
    OPTICAL_FLOW_RUNNING_MEAN,
    OPTICAL_FLOW_WIN_SIZE,
)

__all__ = ["OpticalFlowEstimator"]


class OpticalFlowEstimator:
    """Estima el movimiento global mediante flujo óptico Lucas-Kanade.

    Ubica puntos en una grilla, calcula el flujo entre frames, usa la mediana
    como movimiento crudo y aplica una media móvil para estabilizarlo.

    Args:
        grid_step: Separación en píxeles entre puntos.
        win_size: Tamaño ``(w, h)`` de la ventana Lucas-Kanade.
        max_level: Cantidad de niveles de pirámide.
        running_mean_window: Frames usados para suavizar el movimiento.
    """

    def __init__(
        self,
        grid_step: int = OPTICAL_FLOW_GRID_STEP,
        border_margin: int = OPTICAL_FLOW_BORDER_MARGIN,
        win_size: tuple[int, int] = OPTICAL_FLOW_WIN_SIZE,
        max_level: int = OPTICAL_FLOW_MAX_LEVEL,
        running_mean_window: int = OPTICAL_FLOW_RUNNING_MEAN,
        min_tracking_ratio: float = OPTICAL_FLOW_MIN_TRACKING_RATIO,
    ) -> None:
        self.grid_step = grid_step
        self.border_margin = border_margin
        self.win_size = win_size
        self.max_level = max_level
        self.min_tracking_ratio = min_tracking_ratio

        self._prev_gray: np.ndarray | None = None
        self._motion_history: deque[np.ndarray] = deque(
            maxlen=running_mean_window,
        )
        self.last_tracking_ratio: float = 0.0
        self.last_good_points: int = 0
        self.last_total_points: int = 0
        # OpenCV recibe este diccionario directamente; sus valores se validan
        # al construir el estimador y no se propagan fuera del adaptador.
        self._lk_params: dict[str, object] = {
            "winSize": self.win_size,
            "maxLevel": self.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        }

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Calcula el vector suavizado de movimiento global del frame.

        La primera llamada conserva la imagen en grises y devuelve cero; las
        siguientes devuelven la media móvil del movimiento observado.

        Args:
            frame: Frame BGR según la convención de OpenCV.

        Returns:
            Vector ``[dx, dy]`` del movimiento global suavizado, en píxeles.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self.last_tracking_ratio = 0.0
            self.last_good_points = 0
            self.last_total_points = 0
            return np.zeros(2, dtype=float)

        raw_motion = self._compute_raw_motion(gray)
        self._motion_history.append(raw_motion)
        self._prev_gray = gray

        return np.mean(self._motion_history, axis=0)

    def reset(self) -> None:
        """Descarta el estado temporal entre clips o vistas."""
        self._prev_gray = None
        self._motion_history.clear()
        self.last_tracking_ratio = 0.0
        self.last_good_points = 0
        self.last_total_points = 0

    def _build_grid_points(self, h: int, w: int) -> np.ndarray:
        """Genera puntos de seguimiento sobre una grilla regular.

        Args:
            h: Altura del frame.
            w: Ancho del frame.

        Returns:
            Coordenadas ``float32`` con forma ``(N, 1, 2)``.
        """
        y0 = max(self.border_margin, 0)
        x0 = max(self.border_margin, 0)
        y1 = h - self.border_margin
        x1 = w - self.border_margin

        if x0 >= x1 or y0 >= y1:
            ys = np.arange(0, h, self.grid_step)
            xs = np.arange(0, w, self.grid_step)
        else:
            ys = np.arange(y0, y1, self.grid_step)
            xs = np.arange(x0, x1, self.grid_step)
        grid = np.array(
            np.meshgrid(xs, ys),
            dtype=np.float32,
        ).T.reshape(-1, 1, 2)
        return grid

    def _compute_raw_motion(self, gray: np.ndarray) -> np.ndarray:
        """Ejecuta Lucas-Kanade y devuelve la mediana del desplazamiento.

        Args:
            gray: Frame actual en escala de grises.

        Returns:
            Vector de desplazamiento ``[dx, dy]``.
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
            self.last_tracking_ratio = 0.0
            self.last_good_points = 0
            self.last_total_points = len(pts)
            return np.zeros(2, dtype=float)

        # Sólo los puntos seguidos correctamente participan de la estimación.
        good_mask = status.ravel() == 1
        self.last_total_points = len(pts)
        self.last_good_points = int(np.count_nonzero(good_mask))
        self.last_tracking_ratio = self.last_good_points / max(self.last_total_points, 1)
        if not np.any(good_mask):
            return np.zeros(2, dtype=float)
        if self.last_tracking_ratio < self.min_tracking_ratio:
            return np.zeros(2, dtype=float)

        old_good = pts[good_mask].reshape(-1, 2)
        new_good = new_pts[good_mask].reshape(-1, 2)
        displacements = new_good - old_good

        # La mediana reduce el efecto de vehículos móviles sobre el fondo.
        return np.median(displacements, axis=0)
