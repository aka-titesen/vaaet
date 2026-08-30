# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Public boundary for shared annotated-video analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vaaet.exceptions import VideoOpenError
from vaaet.logging import get_logger
from vaaet.telemetry import CANONICAL_RAW_TELEMETRY_COLUMNS
from vaaet.timestamps import normalize_timestamp
from vaaet.vision.detector import YOLODetector, select_model_variant
from vaaet.vision.hud import HudConfig
from vaaet.vision.optical_flow import OpticalFlowEstimator
from vaaet.vision.pipeline import (
    PipelineMetrics,
    PredictionProvider,
    TrafficStatePrediction,
    VisionPipelineSession,
)
from vaaet.vision.speed import SmoothedSpeedTracker, TrackMotionStateTracker
from vaaet.vision.telemetry import MinuteTelemetryAccumulator
from vaaet.vision.tracking import SORTTracker
from vaaet.vision.video import extract_duration, extract_recording_start, open_video
from vaaet.vision.view_plan import VideoViewPlan, ViewSegmentReport

logger = get_logger(__name__)

__all__ = [
    "PipelineMetrics",
    "PredictionProvider",
    "TrafficStatePrediction",
    "VideoAnalysisResult",
    "VideoViewPlan",
    "ViewSegmentReport",
    "analyze_video",
]

_CLASSIFICATION_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "record_time",
    "traffic_state",
    "state_label",
    "confidence",
    "evidence",
)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Files, tabular outputs and local timings produced by :func:`analyze_video`."""

    video_path: Path
    telemetry: pd.DataFrame
    classifications: pd.DataFrame | None = None
    complete_minutes: int = 0
    processed_duration_seconds: float = 0.0
    discarded_partial_seconds: float = 0.0
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics.empty)
    view_segments: tuple[ViewSegmentReport, ...] = ()


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model_variant: str | None = None,
    prediction_provider: PredictionProvider | None = None,
    hud_config: HudConfig | None = None,
    view_plan: VideoViewPlan | None = None,
    max_frames: int | None = None,
    status_every_seconds: float = 2.0,
) -> VideoAnalysisResult:
    """Analiza un video finito y produce telemetría por minuto más anotaciones.

    Los filtros permanecen síncronos y ordenados. El proveedor opcional agrega
    estado de tráfico al HUD sin acoplar visión a TensorFlow ni a la carga de
    artefactos. El plan de vistas es opt-in y reinicia el estado temporal ante
    cada transición declarada de cámara o encuadre.
    """
    import cv2

    # Retained for API compatibility. Predictions are requested after a complete
    # minute, never on status-panel previews.
    del status_every_seconds

    source = Path(video_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}_vaaet_annotated.mp4")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    duration = extract_duration(str(source))
    selected_variant = model_variant or select_model_variant(duration)
    detector = YOLODetector(model_variant=selected_variant)
    detector.load()
    capture = open_video(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise VideoOpenError(f"Cannot open annotated video writer: {destination}")

    captured_at = extract_recording_start(str(source))
    if captured_at is None:
        captured_at = datetime.now(timezone.utc)
        logger.warning(
            "Filename does not encode capture time; using processing time for %s",
            source.name,
        )
    else:
        captured_at = normalize_timestamp(captured_at).to_pydatetime()

    session = VisionPipelineSession(
        clip_id=source.stem,
        recording_start=captured_at,
        fps=fps,
        frame_height=height,
        frames_per_minute=max(round(fps * 60), 1),
        detector=detector,
        tracker=SORTTracker(),
        flow_estimator=OpticalFlowEstimator(),
        speed_tracker=SmoothedSpeedTracker(window_size=10),
        motion_tracker=TrackMotionStateTracker(),
        accumulator=MinuteTelemetryAccumulator(clip_id=source.stem),
        prediction_provider=prediction_provider,
        hud_config=hud_config or HudConfig(),
        view_plan=view_plan,
    )

    logger.info("Analyzing %s with %s", source.name, selected_variant)
    try:
        output = session.run(capture, writer, max_frames=max_frames)
    finally:
        capture.release()
        writer.release()

    processed_duration_seconds = output.frames_processed / fps
    complete_minutes = len(output.telemetry_records)
    discarded_partial_seconds = (
        processed_duration_seconds % 60.0
        if view_plan is not None
        else max(processed_duration_seconds - complete_minutes * 60.0, 0.0)
    )
    telemetry = pd.DataFrame.from_records(
        output.telemetry_records,
        columns=CANONICAL_RAW_TELEMETRY_COLUMNS,
    )
    classifications = (
        pd.DataFrame.from_records(
            output.classification_records,
            columns=_CLASSIFICATION_COLUMNS,
        )
        if prediction_provider is not None
        else None
    )
    logger.info(
        "Analysis complete: frames=%s telemetry_rows=%s fps=%.2f output=%s",
        output.frames_processed,
        len(telemetry),
        output.metrics.frames_per_second,
        destination,
    )
    return VideoAnalysisResult(
        video_path=destination,
        telemetry=telemetry,
        classifications=classifications,
        complete_minutes=complete_minutes,
        processed_duration_seconds=processed_duration_seconds,
        discarded_partial_seconds=discarded_partial_seconds,
        metrics=output.metrics,
        view_segments=output.view_segments,
    )
