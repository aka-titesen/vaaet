# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Protocolos mínimos para aislar frameworks de inferencia no tipados."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TrafficStateModel(Protocol):
    """Modelo que recibe una matriz escalada y devuelve probabilidades por fila."""

    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray:
        """Devuelve probabilidades de los tres estados aprendidos."""


class FeatureScaler(Protocol):
    """Transformador serializado compatible con el contrato de features v2."""

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transforma una matriz conservando el orden canónico de columnas."""
