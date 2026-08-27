# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Video I/O utilities for the VAAET production pipeline.

Handles strict filename validation (bridge camera naming convention),
duration extraction from filenames, and safe video capture opening.

See ADR-0009 and ADR-0013 for the filename and acquisition decisions.
"""

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
    """Check whether a video filename matches the expected bridge format."""
    basename = os.path.basename(path)
    return bool(_FILENAME_RE.match(basename))


def extract_recording_start(path: str) -> datetime | None:
    """Return the recording start encoded in a bridge filename, if present.

    Free-form filenames deliberately return ``None``. Callers can then use one
    explicit processing timestamp and report the reduced provenance instead of
    silently pretending that processing time is capture time.
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
    """Extract video duration in seconds from a bridge-format filename."""
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
