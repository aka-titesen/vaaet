"""Lightweight calibration helpers for academic speed validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median

import pandas as pd

__all__ = [
    "CalibrationSegment",
    "aggregate_pixels_per_meter",
    "build_calibration_table",
    "pixels_per_meter_from_segment",
    "pseudo_ground_truth_speed_kmh",
]


@dataclass(frozen=True)
class CalibrationSegment:
    """Known bridge landmark segment used for manual calibration."""

    name: str
    pixel_start: tuple[float, float]
    pixel_end: tuple[float, float]
    meters: float
    elapsed_seconds: float | None = None


def _pixel_distance(segment: CalibrationSegment) -> float:
    return hypot(
        float(segment.pixel_end[0]) - float(segment.pixel_start[0]),
        float(segment.pixel_end[1]) - float(segment.pixel_start[1]),
    )


def pixels_per_meter_from_segment(segment: CalibrationSegment) -> float:
    """Return the per-segment pixel-to-meter ratio."""
    if segment.meters <= 0:
        raise ValueError("meters must be > 0")
    px_distance = _pixel_distance(segment)
    if px_distance <= 0:
        raise ValueError("pixel distance must be > 0")
    return px_distance / float(segment.meters)


def aggregate_pixels_per_meter(
    segments: list[CalibrationSegment],
) -> float:
    """Return a robust aggregate calibration ratio from multiple landmarks."""
    if not segments:
        raise ValueError("at least one calibration segment is required")
    ratios = [pixels_per_meter_from_segment(segment) for segment in segments]
    return round(float(median(ratios)), 4)


def pseudo_ground_truth_speed_kmh(
    *,
    distance_m: float,
    elapsed_seconds: float,
) -> float:
    """Convert manual travel time between two landmarks into km/h."""
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be > 0")
    return round((distance_m / elapsed_seconds) * 3.6, 4)


def build_calibration_table(
    segments: list[CalibrationSegment],
) -> pd.DataFrame:
    """Return a notebook-friendly table for manual calibration review."""
    rows: list[dict[str, float | str | None]] = []
    for segment in segments:
        ppm = pixels_per_meter_from_segment(segment)
        speed_kmh = None
        if segment.elapsed_seconds is not None:
            speed_kmh = pseudo_ground_truth_speed_kmh(
                distance_m=segment.meters,
                elapsed_seconds=segment.elapsed_seconds,
            )
        rows.append(
            {
                "name": segment.name,
                "meters": float(segment.meters),
                "pixel_distance": round(_pixel_distance(segment), 4),
                "pixels_per_meter": round(ppm, 4),
                "elapsed_seconds": segment.elapsed_seconds,
                "pseudo_ground_truth_speed_kmh": speed_kmh,
            }
        )
    return pd.DataFrame(rows)
