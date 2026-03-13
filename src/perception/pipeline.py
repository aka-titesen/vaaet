"""Shared telemetry extraction pipeline for the VAAET production notebook."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.config import VEHICLE_TYPES
from src.logging_utils import get_logger
from src.perception.detector import YOLODetector, select_model_variant
from src.perception.optical_flow import OpticalFlowEstimator
from src.perception.speed import (
    SmoothedSpeedTracker,
    TrackMotionStateTracker,
    estimate_speed,
    is_near_zero_motion,
    is_speed_measurement_reliable,
    robust_speed_summary,
)
from src.perception.tracker import SORTTracker, Track
from src.video import extract_duration, open_video, validate_filename

logger = get_logger(__name__)

__all__ = [
    "MinuteTelemetryAccumulator",
    "process_clip_telemetry",
]


def _empty_vehicle_counts() -> dict[str, int]:
    return {vehicle_type: 0 for vehicle_type in VEHICLE_TYPES}


@dataclass
class MinuteTelemetryAccumulator:
    """Accumulate per-minute telemetry, quality, and count-once signals."""

    clip_id: str
    minute_counts: dict[str, int] = field(default_factory=_empty_vehicle_counts)
    minute_speeds: list[float] = field(default_factory=list)
    counted_tracks: set[int] = field(default_factory=set)
    cumulative_counts: dict[str, int] = field(default_factory=_empty_vehicle_counts)
    near_zero_motion_count: int = 0
    stationary_confirmed_count: int = 0
    rejected_speed_count: int = 0
    recovered_track_count: int = 0
    speed_sample_count: int = 0

    def observe_track(
        self,
        track: Track,
        *,
        smoothed_speed: float | None,
        reliable: bool,
        near_zero_motion: bool,
        stationary_confirmed: bool,
        recovered_gap: int,
    ) -> None:
        """Register one track observation into the current minute bucket."""
        if near_zero_motion:
            self.near_zero_motion_count += 1

        if stationary_confirmed:
            self.stationary_confirmed_count += 1
        elif not reliable:
            self.rejected_speed_count += 1
            if recovered_gap > 0:
                self.recovered_track_count += 1
        elif smoothed_speed is not None:
            self.minute_speeds.append(smoothed_speed)
            self.speed_sample_count += 1

        if track.track_id not in self.counted_tracks and track.mark_counted():
            self.minute_counts[track.vehicle_type] = (
                self.minute_counts.get(track.vehicle_type, 0) + 1
            )
            self.counted_tracks.add(track.track_id)

    def has_pending_data(self) -> bool:
        """Return whether the accumulator holds any data worth flushing."""
        return bool(self.minute_speeds or any(self.minute_counts.values()))

    def build_record(self, record_time: datetime) -> dict[str, object]:
        """Materialize the current minute into a telemetry record."""
        total = sum(self.minute_counts.values())
        quality_attempts = self.speed_sample_count + self.rejected_speed_count
        quality = (
            round(self.speed_sample_count / quality_attempts, 4)
            if quality_attempts > 0
            else 0.0
        )
        avg_speed = robust_speed_summary(self.minute_speeds) if self.minute_speeds else 0.0
        return {
            "clip_id": self.clip_id,
            "record_time": record_time,
            "avg_speed": round(avg_speed, 2),
            "count_car": self.minute_counts.get("car", 0),
            "count_truck": self.minute_counts.get("truck", 0),
            "count_bus": self.minute_counts.get("bus", 0),
            "count_motorcycle": self.minute_counts.get("motorcycle", 0),
            "count_bicycle": self.minute_counts.get("bicycle", 0),
            "total_vehicles": total,
            "near_zero_motion_count": self.near_zero_motion_count,
            "stationary_confirmed_count": self.stationary_confirmed_count,
            "rejected_speed_count": self.rejected_speed_count,
            "recovered_track_count": self.recovered_track_count,
            "speed_sample_count": self.speed_sample_count,
            "speed_measurement_quality": quality,
        }

    def rollover_minute(self) -> None:
        """Advance to the next minute and reset minute-local counters."""
        for vehicle_type, count in self.minute_counts.items():
            self.cumulative_counts[vehicle_type] = (
                self.cumulative_counts.get(vehicle_type, 0) + count
            )
        self.minute_counts = _empty_vehicle_counts()
        self.minute_speeds.clear()
        self.counted_tracks.clear()
        self.near_zero_motion_count = 0
        self.stationary_confirmed_count = 0
        self.rejected_speed_count = 0
        self.recovered_track_count = 0
        self.speed_sample_count = 0


def process_clip_telemetry(
    video_path: str,
    *,
    model_variant: str | None = None,
    status_every_seconds: float = 30.0,
) -> pd.DataFrame:
    """Process a clip into per-minute telemetry with robust speed quality."""
    import cv2

    if validate_filename(video_path):
        duration = extract_duration(video_path)
        logger.info("Valid bridge filename detected; duration=%.0fs", duration)
    else:
        logger.info("Non-standard filename detected; falling back to metadata")
        duration = extract_duration(video_path)

    if model_variant is None:
        model_variant = select_model_variant(duration)

    detector = YOLODetector(model_variant=model_variant)
    detector.load()
    tracker = SORTTracker()
    flow_estimator = OpticalFlowEstimator()
    speed_tracker = SmoothedSpeedTracker(window_size=10)
    motion_state_tracker = TrackMotionStateTracker()

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_minute = max(int(fps * 60), 1)
    clip_id = os.path.splitext(os.path.basename(video_path))[0]

    records: list[dict[str, object]] = []
    frame_idx = 0
    accumulator = MinuteTelemetryAccumulator(clip_id=clip_id)
    status_every_frames = max(int(fps * status_every_seconds), 1)

    logger.info(
        "Processing clip %s with %s at %.0f FPS and %s frames",
        clip_id,
        model_variant,
        fps,
        total_frames,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global_motion = flow_estimator.update(frame)
        detections = detector.detect(frame)
        det_tuples = [(d.centroid, d.vehicle_type) for d in detections]
        active_tracks = tracker.update(det_tuples)

        for pruned_track_id in tracker.last_pruned_track_ids:
            speed_tracker.remove_track(pruned_track_id)
            motion_state_tracker.remove_track(pruned_track_id)
            accumulator.counted_tracks.discard(pruned_track_id)

        flow_tracking_ratio = (
            flow_estimator.last_tracking_ratio
            if flow_estimator.last_total_points > 0
            else 1.0
        )

        for track in active_tracks:
            recovered_gap = getattr(track, "recovered_after_gap", 0)
            reliable = is_speed_measurement_reliable(
                track.history,
                flow_tracking_ratio=flow_tracking_ratio,
                recovered_after_gap=recovered_gap,
            )
            speed = None
            if reliable:
                speed = estimate_speed(
                    track.history,
                    fps=fps,
                    frame_height=frame_h,
                    global_motion=global_motion,
                    vehicle_type=track.vehicle_type,
                )

            near_zero_now = is_near_zero_motion(track.history)
            stationary_now = motion_state_tracker.update(
                track.track_id,
                track.history,
                candidate_speed=speed,
            )

            smoothed = None
            if stationary_now or not reliable:
                speed_tracker.remove_track(track.track_id)
            else:
                smoothed = speed_tracker.update(track.track_id, speed)

            accumulator.observe_track(
                track,
                smoothed_speed=smoothed,
                reliable=reliable,
                near_zero_motion=near_zero_now,
                stationary_confirmed=stationary_now,
                recovered_gap=recovered_gap,
            )

        frame_idx += 1

        if frame_idx % status_every_frames == 0:
            avg_speed = robust_speed_summary(accumulator.minute_speeds)
            logger.info(
                "Video time %.0fs | avg_speed=%.1f km/h | total_unique=%s | rejected=%s | stationary=%s",
                frame_idx / fps,
                avg_speed,
                sum(accumulator.cumulative_counts.values()) + sum(accumulator.minute_counts.values()),
                accumulator.rejected_speed_count,
                accumulator.stationary_confirmed_count,
            )

        if frame_idx % frames_per_minute == 0:
            record = accumulator.build_record(datetime.now())
            records.append(record)
            logger.info(
                "Minute %s | avg_speed=%.1f km/h | total_vehicles=%s | quality=%.4f",
                len(records),
                record["avg_speed"],
                record["total_vehicles"],
                record["speed_measurement_quality"],
            )
            accumulator.rollover_minute()

    if frame_idx % frames_per_minute != 0 and accumulator.has_pending_data():
        record = accumulator.build_record(datetime.now())
        records.append(record)
        logger.info(
            "Partial minute %s | avg_speed=%.1f km/h | total_vehicles=%s | quality=%.4f",
            len(records),
            record["avg_speed"],
            record["total_vehicles"],
            record["speed_measurement_quality"],
        )
        accumulator.rollover_minute()

    cap.release()
    logger.info(
        "Processing complete for %s | frames=%s | telemetry_rows=%s",
        clip_id,
        frame_idx,
        len(records),
    )
    return pd.DataFrame(records)
