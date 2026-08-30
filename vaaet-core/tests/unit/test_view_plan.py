# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Pruebas de contratos para planes de vistas y calibración local."""

from __future__ import annotations

from collections import deque

import pytest

from vaaet.exceptions import VideoValidationError
from vaaet.vision.speed import estimate_speed
from vaaet.vision.view_plan import (
    CalibrationReference,
    CameraCalibration,
    VideoViewPlan,
    VideoViewSegment,
)


def _calibration(profile_id: str = "cam-a") -> CameraCalibration:
    return CameraCalibration(
        profile_id=profile_id,
        revision="v1",
        frame_size=(64, 48),
        references=(
            CalibrationReference("far", (0.0, 10.0), (10.0, 10.0), 1.0),
            CalibrationReference("near", (0.0, 40.0), (20.0, 40.0), 1.0),
        ),
    )


def test_calibration_interpolates_scale_by_depth() -> None:
    calibration = _calibration()

    assert calibration.pixels_per_meter_at(10.0) == 10.0
    assert calibration.pixels_per_meter_at(25.0) == 15.0
    assert calibration.pixels_per_meter_at(48.0) == 20.0


def test_calibrated_speed_uses_road_contact_depth() -> None:
    history = deque((index * 10, 10) for index in range(8))

    speed = estimate_speed(
        history,
        fps=10.0,
        frame_height=48,
        vehicle_type="car",
        calibration=_calibration(),
    )

    assert speed == 36.0


def test_plan_from_mapping_resolves_profiles_and_open_final_segment() -> None:
    plan = VideoViewPlan.from_mapping(
        {
            "schema_version": "vaaet-view-plan-v1",
            "profiles": [
                {
                    "profile_id": "cam-a",
                    "revision": "v1",
                    "frame_size": [64, 48],
                    "references": [
                        {
                            "reference_id": "far",
                            "pixel_start": [0, 10],
                            "pixel_end": [10, 10],
                            "meters": 1,
                        },
                        {
                            "reference_id": "near",
                            "pixel_start": [0, 40],
                            "pixel_end": [20, 40],
                            "meters": 1,
                        },
                    ],
                }
            ],
            "segments": [
                {"start_frame": 1, "end_frame": 10, "profile_id": "cam-a"},
                {"start_frame": 10, "end_frame": None, "profile_id": "cam-a"},
            ],
        }
    )

    assert plan.resolve(10, width=64, height=48).profile_id == "cam-a"
    assert plan.segment_index(10) == 1


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        (
            (
                VideoViewSegment(1, 9, "cam-a"),
                VideoViewSegment(10, None, "cam-a"),
            ),
            "contiguous",
        ),
        (
            (VideoViewSegment(2, None, "cam-a"),),
            "frame one",
        ),
    ],
)
def test_plan_rejects_gaps_and_invalid_start(
    segments: tuple[VideoViewSegment, ...],
    message: str,
) -> None:
    with pytest.raises(VideoValidationError, match=message):
        VideoViewPlan(profiles=(_calibration(),), segments=segments)


def test_plan_rejects_resolution_mismatch_without_disclosing_paths() -> None:
    plan = VideoViewPlan(
        profiles=(_calibration(),),
        segments=(VideoViewSegment(1, None, "cam-a"),),
    )

    with pytest.raises(VideoValidationError, match="frame size"):
        plan.resolve(1, width=32, height=48)


def test_plan_rejects_unknown_profiles_and_out_of_bounds_references() -> None:
    with pytest.raises(VideoValidationError, match="unknown profile"):
        VideoViewPlan(
            profiles=(_calibration(),),
            segments=(VideoViewSegment(1, None, "missing-camera"),),
        )

    with pytest.raises(VideoValidationError, match="outside the frame"):
        CameraCalibration(
            profile_id="invalid-camera",
            revision="v1",
            frame_size=(64, 48),
            references=(
                CalibrationReference("far", (0.0, 10.0), (80.0, 10.0), 1.0),
                CalibrationReference("near", (0.0, 40.0), (20.0, 40.0), 1.0),
            ),
        )
