"""Shared annotated-video workflow for telemetry collection and inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from vaaet.data.datasets import RAW_TELEMETRY_V2_COLUMNS
from vaaet.exceptions import VideoOpenError
from vaaet.logging import get_logger
from vaaet.settings import STATE_LABELS
from vaaet.vision.detector import Detection, YOLODetector, select_model_variant
from vaaet.vision.optical_flow import OpticalFlowEstimator
from vaaet.vision.speed import (
    SmoothedSpeedTracker,
    TrackMotionStateTracker,
    estimate_speed,
    is_near_zero_motion,
    is_speed_measurement_reliable,
    robust_speed_summary,
)
from vaaet.vision.telemetry import MinuteTelemetryAccumulator
from vaaet.vision.tracking import SORTTracker, Track
from vaaet.vision.video import (
    extract_duration,
    extract_recording_start,
    open_video,
)

logger = get_logger(__name__)

__all__ = [
    "PredictionProvider",
    "TrafficStatePrediction",
    "VideoAnalysisResult",
    "analyze_video",
]


@dataclass(frozen=True)
class TrafficStatePrediction:
    """Compact state prediction used by the video overlay."""

    state: int
    label: str
    confidence: float
    evidence: float | None = None

    def __post_init__(self) -> None:
        if self.state not in STATE_LABELS:
            raise ValueError(f"Unsupported traffic state: {self.state}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Prediction confidence must be between 0 and 1")


PredictionProvider = Callable[[pd.DataFrame], TrafficStatePrediction | None]


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Files and tabular outputs produced by :func:`analyze_video`."""

    video_path: Path
    telemetry: pd.DataFrame
    classifications: pd.DataFrame | None = None
    complete_minutes: int = 0
    processed_duration_seconds: float = 0.0
    discarded_partial_seconds: float = 0.0


_CLASSIFICATION_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "record_time",
    "traffic_state",
    "state_label",
    "confidence",
    "evidence",
)


_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (80, 210, 80),
    "truck": (40, 150, 255),
    "bus": (80, 190, 255),
    "motorcycle": (255, 170, 30),
    "bicycle": (230, 130, 230),
}

_STATE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (80, 210, 80),
    1: (0, 210, 255),
    2: (0, 120, 255),
    3: (50, 50, 230),
}


def _recording_timestamp(
    recording_start: datetime,
    elapsed_seconds: float,
) -> datetime:
    return recording_start + timedelta(seconds=max(elapsed_seconds, 0.0))


def _nearest_detection(track: Track, detections: list[Detection]) -> Detection | None:
    same_type = [item for item in detections if item.vehicle_type == track.vehicle_type]
    if not same_type:
        return None
    return min(
        same_type,
        key=lambda item: (item.centroid[0] - track.centroid[0]) ** 2
        + (item.centroid[1] - track.centroid[1]) ** 2,
    )


def _draw_track(
    frame: object,
    track: Track,
    detection: Detection | None,
    speed: float | None,
    stationary: bool,
) -> None:
    import cv2

    color = _TYPE_COLORS.get(track.vehicle_type, (220, 220, 220))
    if detection is None:
        cx, cy = track.centroid
        bbox = (cx - 30, cy - 20, cx + 30, cy + 20)
    else:
        bbox = detection.bbox
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    motion = "stationary" if stationary else (f"{speed:.1f} km/h" if speed is not None else "n/a")
    label = f"{track.vehicle_type} #{track.track_id} | {motion}"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_status_panel(
    frame: object,
    accumulator: MinuteTelemetryAccumulator,
    prediction: TrafficStatePrediction | None,
    elapsed_seconds: float,
) -> None:
    import cv2

    height, width = frame.shape[:2]
    panel_width = min(520, max(width - 20, 200))
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 142), (25, 25, 25), -1)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 142), (230, 230, 230), 1)

    total = sum(accumulator.cumulative_counts.values()) + sum(
        accumulator.minute_counts.values()
    )
    avg_speed = robust_speed_summary(accumulator.minute_speeds)
    lines = [
        f"VAAET ML | {elapsed_seconds:,.0f} s",
        f"Vehicles: {total} | Current speed: {avg_speed:.1f} km/h",
        "Counts: "
        + " | ".join(
            f"{kind}={accumulator.cumulative_counts.get(kind, 0) + accumulator.minute_counts.get(kind, 0)}"
            for kind in ("car", "truck", "bus", "motorcycle", "bicycle")
        ),
    ]
    if prediction is None:
        lines.append("Mode: Telemetry Collection")
        state_color = (210, 210, 210)
    else:
        lines.append(f"Traffic: {prediction.label} | confidence={prediction.confidence:.1%}")
        state_color = _STATE_COLORS[prediction.state]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, 38 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            state_color if index == len(lines) - 1 else (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def _prediction_record(
    prediction: TrafficStatePrediction,
    telemetry_record: dict[str, object],
) -> dict[str, object]:
    return {
        "clip_id": telemetry_record["clip_id"],
        "record_time": telemetry_record["record_time"],
        "traffic_state": prediction.state,
        "state_label": prediction.label,
        "confidence": prediction.confidence,
        "evidence": prediction.evidence,
    }


def _request_prediction(
    provider: PredictionProvider | None,
    records: list[dict[str, object]],
    current_record: dict[str, object],
) -> TrafficStatePrediction | None:
    if provider is None:
        return None
    frame = pd.DataFrame([*records, current_record])
    return provider(frame)


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model_variant: str | None = None,
    prediction_provider: PredictionProvider | None = None,
    max_frames: int | None = None,
    status_every_seconds: float = 2.0,
) -> VideoAnalysisResult:
    """Analyze a video once and produce annotations plus minute telemetry.

    The optional provider adds traffic-state overlays without coupling this
    computer-vision pipeline to TensorFlow or the artifact-loading layer.
    """
    import cv2

    # Retained for API compatibility. Predictions are intentionally requested
    # only after a complete minute, never on status-panel previews.
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
    tracker = SORTTracker()
    flow_estimator = OpticalFlowEstimator()
    speed_tracker = SmoothedSpeedTracker(window_size=10)
    motion_tracker = TrackMotionStateTracker()

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

    clip_id = source.stem
    captured_at = extract_recording_start(str(source))
    if captured_at is None:
        captured_at = datetime.now().astimezone().replace(tzinfo=None)
        logger.warning(
            "Filename does not encode capture time; using processing time for %s",
            source.name,
        )

    accumulator = MinuteTelemetryAccumulator(clip_id=clip_id)
    records: list[dict[str, object]] = []
    classification_records: list[dict[str, object]] = []
    latest_prediction: TrafficStatePrediction | None = None
    frames_per_minute = max(round(fps * 60), 1)
    frame_index = 0

    logger.info("Analyzing %s with %s", source.name, selected_variant)
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            global_motion = flow_estimator.update(frame)
            detections = detector.detect(frame)
            tracks = tracker.update([(item.centroid, item.vehicle_type) for item in detections])

            for track_id in tracker.last_pruned_track_ids:
                speed_tracker.remove_track(track_id)
                motion_tracker.remove_track(track_id)
                accumulator.counted_tracks.discard(track_id)

            flow_ratio = (
                flow_estimator.last_tracking_ratio
                if flow_estimator.last_total_points > 0
                else 1.0
            )
            for track in tracks:
                recovered_gap = track.recovered_after_gap
                reliable = is_speed_measurement_reliable(
                    track.history,
                    flow_tracking_ratio=flow_ratio,
                    recovered_after_gap=recovered_gap,
                )
                speed = (
                    estimate_speed(
                        track.history,
                        fps=fps,
                        frame_height=height,
                        global_motion=global_motion,
                        vehicle_type=track.vehicle_type,
                    )
                    if reliable
                    else None
                )
                stationary = motion_tracker.update(
                    track.track_id,
                    track.history,
                    candidate_speed=speed,
                )
                smoothed = None
                if stationary or not reliable:
                    speed_tracker.remove_track(track.track_id)
                else:
                    smoothed = speed_tracker.update(track.track_id, speed)

                accumulator.observe_track(
                    track,
                    smoothed_speed=smoothed,
                    reliable=reliable,
                    near_zero_motion=is_near_zero_motion(track.history),
                    stationary_confirmed=stationary,
                    recovered_gap=recovered_gap,
                    flow_tracking_ratio=flow_ratio,
                )
                _draw_track(
                    frame,
                    track,
                    _nearest_detection(track, detections),
                    smoothed,
                    stationary,
                )

            frame_index += 1
            elapsed = frame_index / fps
            _draw_status_panel(frame, accumulator, latest_prediction, elapsed)
            writer.write(frame)

            if frame_index % frames_per_minute == 0:
                record = accumulator.build_record(_recording_timestamp(captured_at, elapsed))
                records.append(record)
                latest_prediction = _request_prediction(
                    prediction_provider,
                    records[:-1],
                    record,
                )
                if latest_prediction is not None:
                    classification_records.append(_prediction_record(latest_prediction, record))
                accumulator.rollover_minute()

        if frame_index % frames_per_minute:
            logger.info(
                "Discarding final partial minute from classification and telemetry persistence"
            )
    finally:
        capture.release()
        writer.release()

    processed_duration_seconds = frame_index / fps
    complete_minutes = len(records)
    discarded_partial_seconds = max(
        processed_duration_seconds - complete_minutes * 60.0,
        0.0,
    )
    telemetry = pd.DataFrame.from_records(records, columns=RAW_TELEMETRY_V2_COLUMNS)
    classifications = (
        pd.DataFrame.from_records(
            classification_records,
            columns=_CLASSIFICATION_COLUMNS,
        )
        if prediction_provider is not None
        else None
    )
    logger.info(
        "Analysis complete: frames=%s telemetry_rows=%s output=%s",
        frame_index,
        len(telemetry),
        destination,
    )
    return VideoAnalysisResult(
        video_path=destination,
        telemetry=telemetry,
        classifications=classifications,
        complete_minutes=complete_minutes,
        processed_duration_seconds=processed_duration_seconds,
        discarded_partial_seconds=discarded_partial_seconds,
    )
