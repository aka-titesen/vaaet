# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Excepciones de dominio portables para artefactos y visión."""

from __future__ import annotations


class VAAETError(Exception):
    """Raíz de los errores de dominio portables de VAAET."""


class ArtifactNotFoundError(FileNotFoundError, VAAETError):
    """Señala que falta un artefacto local requerido por un contrato."""


class ArtifactValidationError(ValueError, VAAETError):
    """Señala que un bundle o artefacto viola su contrato portable."""


class VideoValidationError(ValueError, VAAETError):
    """Señala que un video no puede validarse de manera segura."""


class VideoOpenError(RuntimeError, VAAETError):
    """Señala que OpenCV no pudo abrir un recurso de video."""
