"""Tests for the shared annotated-video workflow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vaaet.data.datasets import RAW_TELEMETRY_V2_COLUMNS
from vaaet.vision import analysis
from vaaet.vision.analysis import TrafficStatePrediction, analyze_video
from vaaet.vision.detector import Detection


class _FakeDetector:
    def __init__(self, **_: object) -> None:
        self.frame = 0

    def load(self) -> None:
        return None

    def detect(self, _frame: object) -> list[Detection]:
        self.frame += 1
        x = 20 + self.frame * 3
        return [Detection((x, 30, x + 20, 50), "car", 0.9)]


class _FakeFlow:
    last_tracking_ratio = 1.0
    last_total_points = 10

    def update(self, _frame: object) -> np.ndarray:
        return np.zeros(2, dtype=float)


def _make_video(path: Path, *, frames: int = 60) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (96, 64))
    for _ in range(frames):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()


def test_analyze_video_with_and_without_prediction_provider(tmp_path, monkeypatch) -> None:
    source = tmp_path / "bridge_2025-05-01_08-00-00_to_08-01-00.mp4"
    _make_video(source)
    monkeypatch.setattr(analysis, "YOLODetector", _FakeDetector)
    monkeypatch.setattr(analysis, "OpticalFlowEstimator", _FakeFlow)

    collection = analyze_video(source, tmp_path / "collection.mp4", max_frames=60)
    assert collection.video_path.is_file()
    assert not collection.telemetry.empty
    assert collection.classifications is None
    assert collection.complete_minutes == 1
    assert collection.discarded_partial_seconds == 0.0

    provider_calls = 0

    def provider(_telemetry: object) -> TrafficStatePrediction:
        nonlocal provider_calls
        provider_calls += 1
        return TrafficStatePrediction(0, "Normal", 0.9)

    inference = analyze_video(
        source,
        tmp_path / "inference.mp4",
        prediction_provider=provider,
        max_frames=60,
        status_every_seconds=0.2,
    )
    assert inference.video_path.is_file()
    assert inference.classifications is not None
    assert inference.classifications.iloc[-1]["state_label"] == "Normal"
    assert provider_calls == 1


def test_analyze_video_returns_canonical_empty_frames_for_short_clip(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "bridge_2025-05-01_08-00-00_to_08-00-16.mp4"
    _make_video(source, frames=16)
    monkeypatch.setattr(analysis, "YOLODetector", _FakeDetector)
    monkeypatch.setattr(analysis, "OpticalFlowEstimator", _FakeFlow)

    provider_calls = 0

    def provider(_telemetry: object) -> TrafficStatePrediction:
        nonlocal provider_calls
        provider_calls += 1
        return TrafficStatePrediction(0, "Normal", 0.9)

    result = analyze_video(
        source,
        tmp_path / "short.mp4",
        prediction_provider=provider,
    )

    assert result.video_path.is_file()
    assert result.telemetry.empty
    assert tuple(result.telemetry.columns) == RAW_TELEMETRY_V2_COLUMNS
    assert result.classifications is not None
    assert result.classifications.empty
    assert tuple(result.classifications.columns) == (
        "clip_id",
        "record_time",
        "traffic_state",
        "state_label",
        "confidence",
        "evidence",
    )
    assert result.complete_minutes == 0
    assert result.processed_duration_seconds == 16.0
    assert result.discarded_partial_seconds == 16.0
    assert provider_calls == 0


def test_analyze_video_discards_only_the_partial_tail(tmp_path, monkeypatch) -> None:
    source = tmp_path / "bridge_2025-05-01_08-00-00_to_08-02-05.mp4"
    _make_video(source, frames=125)
    monkeypatch.setattr(analysis, "YOLODetector", _FakeDetector)
    monkeypatch.setattr(analysis, "OpticalFlowEstimator", _FakeFlow)

    result = analyze_video(source, tmp_path / "partial-tail.mp4")

    assert len(result.telemetry) == 2
    assert result.complete_minutes == 2
    assert result.processed_duration_seconds == 125.0
    assert result.discarded_partial_seconds == 5.0
