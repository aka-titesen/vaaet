# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for vaaet.vision.video — filename validation, duration extraction, video opening."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from vaaet.vision.video import (
    extract_duration,
    extract_recording_start,
    open_video,
    validate_filename,
)


def test_extract_recording_start_from_bridge_filename() -> None:
    result = extract_recording_start("bridge_2024-01-15_23-55-00_to_00-05-00.mp4")
    assert result is not None
    assert result.isoformat() == "2024-01-15T23:55:00"


def test_extract_recording_start_rejects_free_form_filename() -> None:
    assert extract_recording_start("camera-export.mp4") is None


class TestValidateFilename:
    """Tests for :func:`validate_filename`."""

    def test_valid_standard_filename(self) -> None:
        assert validate_filename("bridge_2024-01-15_08-00-00_to_08-05-00.mp4") is True

    def test_valid_with_path(self) -> None:
        assert (
            validate_filename("/data/videos/bridge_2024-01-15_08-00-00_to_08-05-00.mp4")
            is True
        )

    def test_invalid_no_bridge_prefix(self) -> None:
        assert validate_filename("video_2024-01-15_08-00-00_to_08-05-00.mp4") is False

    def test_invalid_wrong_extension(self) -> None:
        assert validate_filename("bridge_2024-01-15_08-00-00_to_08-05-00.avi") is False

    def test_invalid_missing_timestamps(self) -> None:
        assert validate_filename("bridge_2024-01-15.mp4") is False

    def test_invalid_empty_string(self) -> None:
        assert validate_filename("") is False

    def test_invalid_random_name(self) -> None:
        assert validate_filename("my_cool_video.mp4") is False


class TestExtractDuration:
    """Tests for :func:`extract_duration`."""

    def test_five_minute_clip(self) -> None:
        path = "bridge_2024-01-15_08-00-00_to_08-05-00.mp4"
        assert extract_duration(path) == 300.0  # 5 minutes

    def test_one_hour_clip(self) -> None:
        path = "bridge_2024-01-15_08-00-00_to_09-00-00.mp4"
        assert extract_duration(path) == 3600.0  # 1 hour

    def test_thirty_second_clip(self) -> None:
        path = "bridge_2024-01-15_14-30-00_to_14-30-30.mp4"
        assert extract_duration(path) == 30.0

    def test_midnight_crossing(self) -> None:
        """End time < start time → assumed to cross midnight."""
        path = "bridge_2024-01-15_23-55-00_to_00-05-00.mp4"
        duration = extract_duration(path)
        assert duration == 600.0  # 10 minutes

    def test_non_standard_filename_with_metadata(self) -> None:
        """Non-standard filename falls back to metadata read."""
        # Create a tiny temp video (~1 frame)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (100, 100),
            )
            for _ in range(30):  # 1 second at 30fps
                writer.write(frame)
            writer.release()

            duration = extract_duration(tmp_path)
            assert 0.5 < duration < 2.0  # ~1 second
        finally:
            os.unlink(tmp_path)


class TestOpenVideo:
    """Tests for :func:`open_video`."""

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            open_video("/nonexistent/path/video.mp4")

    def test_valid_video_opens(self) -> None:
        """Create a minimal video and verify it opens."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (100, 100),
            )
            writer.write(frame)
            writer.release()

            cap = open_video(tmp_path)
            assert cap.isOpened()
            cap.release()
        finally:
            os.unlink(tmp_path)
