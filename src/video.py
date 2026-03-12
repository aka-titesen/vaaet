"""Video I/O utilities for the VAAET production pipeline.

Handles strict filename validation (bridge camera naming convention),
duration extraction from filenames, and safe video capture opening.

References:
    - Legacy: ``VAAETHybrid.validate_filename()`` and
      ``extract_duration_from_filename()`` in ``archive/00_bootstrap/``
    - ADR-009 §Video I/O
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import cv2

from src.config import VIDEO_FILENAME_PATTERN
from src.exceptions import ArtifactNotFoundError, VideoOpenError, VideoValidationError

__all__ = [
    "validate_filename",
    "extract_duration",
    "open_video",
]

_FILENAME_RE = re.compile(VIDEO_FILENAME_PATTERN)


def validate_filename(path: str) -> bool:
    """Check whether a video filename matches the expected bridge format."""
    basename = os.path.basename(path)
    return bool(_FILENAME_RE.match(basename))


def extract_duration(path: str) -> float:
    """Extract video duration in seconds from a bridge-format filename."""
    basename = os.path.basename(path)
    match = _FILENAME_RE.match(basename)
    if match:
        parts = basename.replace(".mp4", "").split("_")
        date_str = parts[1]
        start_time_str = parts[2]
        end_time_str = parts[4]

        start_dt = datetime.strptime(
            f"{date_str} {start_time_str.replace('-', ':')}",
            "%Y-%m-%d %H:%M:%S",
        )
        end_dt = datetime.strptime(
            f"{date_str} {end_time_str.replace('-', ':')}",
            "%Y-%m-%d %H:%M:%S",
        )

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return (end_dt - start_dt).total_seconds()

    return _duration_from_metadata(path)


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and return a ``cv2.VideoCapture`` with validation."""
    if not os.path.isfile(path):
        raise ArtifactNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {path}")

    return cap


def _duration_from_metadata(path: str) -> float:
    """Read duration from video file using OpenCV metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoValidationError(f"Cannot open video for metadata: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if frame_count <= 0:
        raise VideoValidationError(f"Cannot determine frame count: {path}")

    return frame_count / fps
