# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Utilidades de logging sin efectos globales al importar librerías."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el logger raíz desde un entrypoint o notebook explícito."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=_DEFAULT_FORMAT,
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger nombrado sin modificar la configuración global."""
    return logging.getLogger(name)
