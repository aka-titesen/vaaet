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
from typing import Any

import cv2

from src.config import VIDEO_FILENAME_PATTERN

__all__ = [
    "validate_filename",
    "extract_duration",
    "open_video",
]

# Pre-compiled regex for performance
_FILENAME_RE = re.compile(VIDEO_FILENAME_PATTERN)


def validate_filename(path: str) -> bool:
    """Check whether a video filename matches the expected bridge format.

    Expected pattern: ``bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4``

    Args:
        path: File path (only the basename is validated).

    Returns:
        ``True`` if the filename is valid.
    """
    basename = os.path.basename(path)
    return bool(_FILENAME_RE.match(basename))


def extract_duration(path: str) -> float:
    """Extract video duration in seconds from a bridge-format filename.

    Parses the start/end timestamps embedded in the filename:
    ``bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4``

    Falls back to reading the duration from the video file itself if the
    filename doesn't match the expected pattern.

    Args:
        path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the filename is non-standard AND the file cannot
            be opened for metadata inspection.
    """
    basename = os.path.basename(path)

    # Try filename-based extraction first
    match = _FILENAME_RE.match(basename)
    if match:
        # Parse timestamps from the filename
        parts = basename.replace(".mp4", "").split("_")
        # parts: ['bridge', 'YYYY-MM-DD', 'HH-MM-SS', 'to', 'HH-MM-SS']
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

        # Handle midnight crossing
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return (end_dt - start_dt).total_seconds()

    # Fallback: read duration from video metadata
    return _duration_from_metadata(path)


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and return a ``cv2.VideoCapture`` with validation.

    Args:
        path: Path to the video file.

    Returns:
        An opened ``cv2.VideoCapture`` object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    return cap


# Private helpers


def _duration_from_metadata(path: str) -> float:
    """Read duration from video file using OpenCV metadata.

    Args:
        path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for metadata: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Cannot determine frame count: {path}")

    return frame_count / fps
