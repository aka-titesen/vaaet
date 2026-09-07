# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for synchronous Pipe-and-Filter vision internals."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from vaaet.exceptions import VideoValidationError
from vaaet.vision.detector import Detection
from vaaet.vision.hud import HudConfig
from vaaet.vision.pipeline import (
    FramePacket,
    PerceptionPacket,
    PipelineMetricsCollector,
    VisionPipelineSession,
)
from vaaet.vision.speed import SmoothedSpeedTracker, TrackMotionStateTracker
from vaaet.vision.telemetry import MinuteTelemetryAccumulator
from vaaet.vision.tracking import SORTTracker
from vaaet.vision.view_plan import (
    CalibrationReference,
    CameraCalibration,
    VideoViewPlan,
    VideoViewSegment,
)


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        x = self.calls * 4
        return [Detection((x, 10, x + 12, 24), "car", 0.9)]


class _FakeFlow:
    last_tracking_ratio = 1.0
    last_total_points = 8

    def update(self, _frame: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.reset_calls = getattr(self, "reset_calls", 0) + 1


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)

    def read(self) -> tuple[bool, np.ndarray]:
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, np.empty((0, 0, 3), dtype=np.uint8)


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


def _session() -> VisionPipelineSession:
    return VisionPipelineSession(
        clip_id="test-clip",
        recording_start=datetime(2025, 5, 1, tzinfo=timezone.utc),
        fps=1.0,
        frame_height=48,
        frames_per_minute=2,
        detector=_FakeDetector(),
        tracker=SORTTracker(max_lost=0),
        flow_estimator=_FakeFlow(),  # type: ignore[arg-type]
        speed_tracker=SmoothedSpeedTracker(),
        motion_tracker=TrackMotionStateTracker(),
        accumulator=MinuteTelemetryAccumulator(clip_id="test-clip"),
        prediction_provider=None,
        hud_config=HudConfig(),
        clock=lambda: 1.0,
    )


def test_metrics_collector_uses_injected_monotonic_clock() -> None:
    collector = PipelineMetricsCollector(clock=_FakeClock([0.0, 1.0, 3.0, 4.0]))

    assert collector.measure("read", lambda: "frame") == "frame"

    metrics = collector.finish(frames_processed=2)
    assert metrics.processing_seconds == 4.0
    assert metrics.frames_per_second == 0.5
    assert metrics.stage_seconds["read"] == 2.0
    assert metrics.stage_seconds["tracking"] == 0.0


def test_frame_packet_rejects_malformed_input_with_stage_name() -> None:
    with pytest.raises(VideoValidationError, match="vision.frame_read"):
        FramePacket(
            clip_id="test-clip",
            frame_index=0,
            fps=1.0,
            elapsed_seconds=0.0,
            capture_time=datetime(2025, 5, 1, tzinfo=timezone.utc),
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )


def test_perception_packet_rejects_invalid_motion_with_stage_name() -> None:
    frame = FramePacket(
        clip_id="test-clip",
        frame_index=1,
        fps=1.0,
        elapsed_seconds=1.0,
        capture_time=datetime(2025, 5, 1, tzinfo=timezone.utc),
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    with pytest.raises(VideoValidationError, match="vision.perception"):
        PerceptionPacket(
            frame=frame,
            global_motion=np.zeros(3, dtype=float),
            flow_tracking_ratio=1.0,
            detections=(),
        )


def test_session_preserves_order_and_materializes_complete_minutes() -> None:
    session = _session()
    writer = _FakeWriter()
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]

    output = session.run(_FakeCapture(frames), writer, max_frames=None)

    assert output.frames_processed == 2
    assert len(writer.frames) == 2
    assert len(output.telemetry_records) == 1
    assert output.telemetry_records[0]["record_time"] == datetime(
        2025,
        5,
        1,
        0,
        0,
        2,
        tzinfo=timezone.utc,
    )
    assert output.metrics.frames_processed == 2


def test_session_rejects_out_of_order_packets() -> None:
    session = _session()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    packet = FramePacket(
        clip_id="test-clip",
        frame_index=2,
        fps=1.0,
        elapsed_seconds=2.0,
        capture_time=datetime(2025, 5, 1, tzinfo=timezone.utc),
        frame=frame,
    )

    with pytest.raises(VideoValidationError, match="vision.frame_order"):
        session.process_frame(packet)


def test_view_transition_resets_state_and_discards_crossed_minute() -> None:
    calibration = CameraCalibration(
        profile_id="cam-a",
        revision="v1",
        frame_size=(64, 48),
        references=(
            CalibrationReference("far", (0.0, 10.0), (10.0, 10.0), 1.0),
            CalibrationReference("near", (0.0, 40.0), (20.0, 40.0), 1.0),
        ),
    )
    plan = VideoViewPlan(
        profiles=(calibration,),
        segments=(
            VideoViewSegment(1, 3, "cam-a"),
            VideoViewSegment(3, None, "cam-a"),
        ),
    )
    provider_calls = 0

    def provider(_telemetry: object) -> None:
        nonlocal provider_calls
        provider_calls += 1
        return None

    session = VisionPipelineSession(
        clip_id="test-clip",
        recording_start=datetime(2025, 5, 1, tzinfo=timezone.utc),
        fps=1.0,
        frame_height=48,
        frames_per_minute=4,
        detector=_FakeDetector(),
        tracker=SORTTracker(max_lost=0),
        flow_estimator=_FakeFlow(),  # type: ignore[arg-type]
        speed_tracker=SmoothedSpeedTracker(),
        motion_tracker=TrackMotionStateTracker(),
        accumulator=MinuteTelemetryAccumulator(clip_id="test-clip"),
        prediction_provider=provider,
        hud_config=HudConfig(),
        view_plan=plan,
        clock=lambda: 1.0,
    )

    output = session.run(
        _FakeCapture([np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(12)]),
        _FakeWriter(),
        max_frames=None,
    )

    assert len(output.telemetry_records) == 2
    assert provider_calls == 1
    assert len(output.view_segments) == 2
    assert output.view_segments[1].discarded_minutes == 1
    assert output.view_segments[1].discard_reason == "transition_crossed_minute"
    assert isinstance(session.flow_estimator, _FakeFlow)
    assert session.flow_estimator.reset_calls == 1
    assert len(session.tracker.active_tracks[0].history) == 10


def test_view_transition_on_minute_boundary_starts_a_clean_continuity() -> None:
    calibration = CameraCalibration(
        profile_id="cam-a",
        revision="v1",
        frame_size=(64, 48),
        references=(
            CalibrationReference("far", (0.0, 10.0), (10.0, 10.0), 1.0),
            CalibrationReference("near", (0.0, 40.0), (20.0, 40.0), 1.0),
        ),
    )
    plan = VideoViewPlan(
        profiles=(calibration,),
        segments=(VideoViewSegment(1, 5, "cam-a"), VideoViewSegment(5, None, "cam-a")),
    )
    session = VisionPipelineSession(
        clip_id="boundary-clip",
        recording_start=datetime(2025, 5, 1, tzinfo=timezone.utc),
        fps=1.0,
        frame_height=48,
        frames_per_minute=4,
        detector=_FakeDetector(),
        tracker=SORTTracker(max_lost=0),
        flow_estimator=_FakeFlow(),  # type: ignore[arg-type]
        speed_tracker=SmoothedSpeedTracker(),
        motion_tracker=TrackMotionStateTracker(),
        accumulator=MinuteTelemetryAccumulator(clip_id="boundary-clip"),
        prediction_provider=None,
        hud_config=HudConfig(),
        view_plan=plan,
        clock=lambda: 1.0,
    )

    output = session.run(
        _FakeCapture([np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(8)]),
        _FakeWriter(),
        max_frames=None,
    )

    assert len(output.telemetry_records) == 2
    assert [row["continuity_id"] for row in output.telemetry_records] == [
        "boundary-clip:continuity-0001",
        "boundary-clip:continuity-0002",
    ]
    assert sum(segment.discarded_minutes for segment in output.view_segments) == 0
