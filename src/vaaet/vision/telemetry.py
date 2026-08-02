"""Minute-level telemetry accumulation for VAAET video workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vaaet.settings import VEHICLE_TYPES
from vaaet.vision.speed import robust_speed_summary
from vaaet.vision.tracking import Track

__all__ = ["MinuteTelemetryAccumulator"]


def _empty_vehicle_counts() -> dict[str, int]:
    return {vehicle_type: 0 for vehicle_type in VEHICLE_TYPES}


@dataclass
class MinuteTelemetryAccumulator:
    """Accumulate per-minute telemetry, quality, and count-once signals."""

    clip_id: str
    minute_counts: dict[str, int] = field(default_factory=_empty_vehicle_counts)
    minute_speeds: list[float] = field(default_factory=list)
    flow_tracking_ratios: list[float] = field(default_factory=list)
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
        flow_tracking_ratio: float,
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

        self.flow_tracking_ratios.append(flow_tracking_ratio)

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
        avg_flow = (
            sum(self.flow_tracking_ratios) / len(self.flow_tracking_ratios)
            if self.flow_tracking_ratios
            else 1.0
        )

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
            "optical_flow_tracking_ratio": round(avg_flow, 4),
        }

    def rollover_minute(self) -> None:
        """Advance to the next minute and reset minute-local counters."""
        for vehicle_type, count in self.minute_counts.items():
            self.cumulative_counts[vehicle_type] = (
                self.cumulative_counts.get(vehicle_type, 0) + count
            )
        self.minute_counts = _empty_vehicle_counts()
        self.minute_speeds.clear()
        self.flow_tracking_ratios.clear()
        self.counted_tracks.clear()
        self.near_zero_motion_count = 0
        self.stationary_confirmed_count = 0
        self.rejected_speed_count = 0
        self.recovered_track_count = 0
        self.speed_sample_count = 0
