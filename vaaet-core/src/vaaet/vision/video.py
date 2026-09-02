# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Utilidades de I/O y metadatos para videos del pipeline VAAET."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import cv2

from vaaet.exceptions import ArtifactNotFoundError, VideoOpenError, VideoValidationError
from vaaet.settings import VIDEO_FILENAME_PATTERN

__all__ = [
    "validate_filename",
    "extract_duration",
    "extract_recording_start",
    "open_video",
]

_FILENAME_RE = re.compile(VIDEO_FILENAME_PATTERN)


def validate_filename(path: str) -> bool:
    """Verifica el formato contractual del nombre de un video."""
    basename = os.path.basename(path)
    return bool(_FILENAME_RE.match(basename))


def extract_recording_start(path: str) -> datetime | None:
    """Extrae el inicio de grabación codificado en el nombre, si existe.

    Los nombres libres devuelven ``None`` para que el consumidor declare un
    timestamp de procesamiento sin presentarlo como tiempo de captura.
    """
    basename = os.path.basename(path)
    match = _FILENAME_RE.match(basename)
    if not match:
        return None

    parts = basename.removesuffix(".mp4").split("_")
    return datetime.strptime(
        f"{parts[1]} {parts[2].replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    )


def extract_duration(path: str) -> float:
    """Extrae la duración en segundos desde el nombre o los metadatos."""
    basename = os.path.basename(path)
    match = _FILENAME_RE.match(basename)
    if match:
        parts = basename.replace(".mp4", "").split("_")
        date_str = parts[1]
        end_time_str = parts[4]

        start_dt = extract_recording_start(path)
        assert start_dt is not None
        end_dt = datetime.strptime(
            f"{date_str} {end_time_str.replace('-', ':')}",
            "%Y-%m-%d %H:%M:%S",
        )

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return (end_dt - start_dt).total_seconds()

    return _duration_from_metadata(path)


def open_video(path: str) -> cv2.VideoCapture:
    """Abre un video y valida el ``cv2.VideoCapture`` resultante."""
    if not os.path.isfile(path):
        raise ArtifactNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {path}")

    return cap


def _duration_from_metadata(path: str) -> float:
    """Calcula la duración mediante metadatos de OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoValidationError(f"Cannot open video for metadata: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if frame_count <= 0:
        raise VideoValidationError(f"Cannot determine frame count: {path}")

    return frame_count / fps
