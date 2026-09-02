# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Acumulación de telemetría por minuto para workflows de video."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vaaet.settings import TELEMETRY_SCHEMA_VERSION, VEHICLE_TYPES
from vaaet.vision.speed import robust_speed_summary
from vaaet.vision.tracking import Track

__all__ = ["MinuteTelemetryAccumulator"]


def _empty_vehicle_counts() -> dict[str, int]:
    return {vehicle_type: 0 for vehicle_type in VEHICLE_TYPES}


@dataclass
class MinuteTelemetryAccumulator:
    """Acumula por minuto telemetría, calidad y conteos únicos."""

    clip_id: str
    minute_counts: dict[str, int] = field(default_factory=_empty_vehicle_counts)
    minute_speeds: list[float] = field(default_factory=list)
    flow_tracking_ratios: list[float] = field(default_factory=list)
    counted_tracks: set[int] = field(default_factory=set)
    observed_track_ids: set[int] = field(default_factory=set)
    near_zero_track_ids: set[int] = field(default_factory=set)
    stationary_track_ids: set[int] = field(default_factory=set)
    rejected_speed_track_ids: set[int] = field(default_factory=set)
    recovered_track_ids: set[int] = field(default_factory=set)
    reliable_speed_track_ids: set[int] = field(default_factory=set)
    cumulative_counts: dict[str, int] = field(default_factory=_empty_vehicle_counts)

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
        """Registra la observación de un track en el minuto actual."""
        track_id = track.track_id
        self.observed_track_ids.add(track_id)
        if near_zero_motion:
            self.near_zero_track_ids.add(track_id)

        if stationary_confirmed:
            self.stationary_track_ids.add(track_id)
        elif not reliable:
            self.rejected_speed_track_ids.add(track_id)
            if recovered_gap > 0:
                self.recovered_track_ids.add(track_id)
        elif smoothed_speed is not None:
            self.minute_speeds.append(smoothed_speed)
            self.reliable_speed_track_ids.add(track_id)

        self.flow_tracking_ratios.append(flow_tracking_ratio)

        if track.track_id not in self.counted_tracks and track.mark_counted():
            self.minute_counts[track.vehicle_type] = (
                self.minute_counts.get(track.vehicle_type, 0) + 1
            )
            self.counted_tracks.add(track.track_id)

    def has_pending_data(self) -> bool:
        """Indica si el minuto contiene información para materializar."""
        return bool(self.observed_track_ids or self.minute_speeds or any(self.minute_counts.values()))

    def build_record(self, record_time: datetime) -> dict[str, object]:
        """Materializa el minuto actual como registro de telemetría."""
        total = sum(self.minute_counts.values())
        speed_sample_count = len(self.reliable_speed_track_ids)
        rejected_ids = self.rejected_speed_track_ids - self.reliable_speed_track_ids
        rejected_speed_count = len(rejected_ids)
        quality_attempts = speed_sample_count + rejected_speed_count
        quality = (
            round(speed_sample_count / quality_attempts, 4)
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
            "near_zero_motion_count": len(self.near_zero_track_ids),
            "stationary_confirmed_count": len(self.stationary_track_ids),
            "rejected_speed_count": rejected_speed_count,
            "recovered_track_count": len(self.recovered_track_ids),
            "speed_sample_count": speed_sample_count,
            "speed_measurement_quality": quality,
            "optical_flow_tracking_ratio": round(avg_flow, 4),
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        }

    def rollover_minute(self) -> None:
        """Avanza al minuto siguiente y reinicia sus contadores locales."""
        for vehicle_type, count in self.minute_counts.items():
            self.cumulative_counts[vehicle_type] = (
                self.cumulative_counts.get(vehicle_type, 0) + count
            )
        self._clear_minute()

    def discard_minute(self) -> None:
        """Descarta un minuto mezclado entre vistas sin afectar acumulados del HUD."""
        self._clear_minute()

    def _clear_minute(self) -> None:
        """Restablece exclusivamente las señales locales del minuto en curso."""
        self.minute_counts = _empty_vehicle_counts()
        self.minute_speeds.clear()
        self.flow_tracking_ratios.clear()
        self.counted_tracks.clear()
        self.observed_track_ids.clear()
        self.near_zero_track_ids.clear()
        self.stationary_track_ids.clear()
        self.rejected_speed_track_ids.clear()
        self.recovered_track_ids.clear()
        self.reliable_speed_track_ids.clear()
