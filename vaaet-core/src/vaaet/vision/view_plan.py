# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contratos de calibración explícita para vistas estables de un video."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median
from typing import cast

from vaaet.exceptions import VideoValidationError

__all__ = [
    "CalibrationReference",
    "CameraCalibration",
    "VideoViewPlan",
    "VideoViewSegment",
    "ViewSegmentReport",
]

_PLAN_SCHEMA_VERSION = "vaaet-view-plan-v1"
_Point = tuple[float, float]


@dataclass(frozen=True)
class CalibrationReference:
    """Tramo vial de longitud conocida usado para calibrar una vista estable."""

    reference_id: str
    pixel_start: _Point
    pixel_end: _Point
    meters: float

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise VideoValidationError("vision.calibration.reference: id is required")
        if not isfinite(self.meters) or self.meters <= 0:
            raise VideoValidationError(
                "vision.calibration.reference: meters must be a finite positive value"
            )
        if not all(isfinite(value) for point in (self.pixel_start, self.pixel_end) for value in point):
            raise VideoValidationError(
                "vision.calibration.reference: pixel coordinates must be finite"
            )
        if self.pixel_length <= 0:
            raise VideoValidationError(
                "vision.calibration.reference: pixel reference must have length"
            )

    @property
    def pixel_length(self) -> float:
        """Devuelve la distancia de la referencia en píxeles."""
        return hypot(
            self.pixel_end[0] - self.pixel_start[0],
            self.pixel_end[1] - self.pixel_start[1],
        )

    @property
    def pixels_per_meter(self) -> float:
        """Devuelve la escala métrica local derivada de la referencia."""
        return self.pixel_length / self.meters

    @property
    def depth_y(self) -> float:
        """Devuelve la profundidad aproximada por el punto medio vertical."""
        return (self.pixel_start[1] + self.pixel_end[1]) / 2.0


@dataclass(frozen=True)
class CameraCalibration:
    """Calibración métrica versionada para una cámara y encuadre estables."""

    profile_id: str
    revision: str
    frame_size: tuple[int, int]
    references: tuple[CalibrationReference, ...]

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if not self.profile_id or not self.revision:
            raise VideoValidationError("vision.calibration.profile: id and revision are required")
        if width < 1 or height < 1:
            raise VideoValidationError("vision.calibration.profile: frame_size must be positive")
        if len(self.references) < 2:
            raise VideoValidationError(
                "vision.calibration.profile: at least two references are required"
            )
        reference_ids = [reference.reference_id for reference in self.references]
        if len(set(reference_ids)) != len(reference_ids):
            raise VideoValidationError("vision.calibration.profile: reference ids must be unique")
        for reference in self.references:
            self._validate_reference_bounds(reference)
        if len({reference.depth_y for reference in self.references}) < 2:
            raise VideoValidationError(
                "vision.calibration.profile: references must cover two image depths"
            )

    def _validate_reference_bounds(self, reference: CalibrationReference) -> None:
        width, height = self.frame_size
        for x, y in (reference.pixel_start, reference.pixel_end):
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise VideoValidationError(
                    "vision.calibration.profile: reference lies outside the frame"
                )

    def validate_frame_size(self, width: int, height: int) -> None:
        """Rechaza una calibración aplicada a una resolución distinta."""
        if (width, height) != self.frame_size:
            raise VideoValidationError(
                "vision.calibration.profile: frame size does not match the calibration"
            )

    def pixels_per_meter_at(self, y: float) -> float:
        """Interpola la escala local por profundidad y limita fuera de los extremos."""
        scales_by_depth: dict[float, list[float]] = {}
        for reference in self.references:
            scales_by_depth.setdefault(reference.depth_y, []).append(reference.pixels_per_meter)
        scale_points = sorted(
            (depth, float(median(scales))) for depth, scales in scales_by_depth.items()
        )
        first_depth, first_scale = scale_points[0]
        last_depth, last_scale = scale_points[-1]
        if y <= first_depth:
            return first_scale
        if y >= last_depth:
            return last_scale
        for (left_depth, left_scale), (right_depth, right_scale) in zip(
            scale_points, scale_points[1:], strict=False
        ):
            if left_depth <= y <= right_depth:
                ratio = (y - left_depth) / (right_depth - left_depth)
                return left_scale + (right_scale - left_scale) * ratio
        raise VideoValidationError("vision.calibration.profile: cannot resolve local scale")

    def distance_meters(
        self,
        positions: Sequence[_Point],
        displacements: Sequence[_Point],
    ) -> float:
        """Convierte desplazamientos consecutivos a metros usando su profundidad local."""
        if len(positions) != len(displacements) + 1:
            raise VideoValidationError(
                "vision.calibration.profile: positions and displacements are inconsistent"
            )
        return sum(
            hypot(displacement[0], displacement[1]) / self.pixels_per_meter_at(position[1])
            for position, displacement in zip(positions[1:], displacements, strict=False)
        )


@dataclass(frozen=True)
class VideoViewSegment:
    """Rango de frames asociado a un perfil de cámara preaprobado."""

    start_frame: int
    end_frame: int | None
    profile_id: str

    def __post_init__(self) -> None:
        if self.start_frame < 1:
            raise VideoValidationError("vision.view_plan.segment: start_frame must be at least one")
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise VideoValidationError(
                "vision.view_plan.segment: end_frame must be greater than start_frame"
            )
        if not self.profile_id:
            raise VideoValidationError("vision.view_plan.segment: profile_id is required")

    def contains(self, frame_index: int) -> bool:
        """Indica si un frame pertenece al rango semiabierto del segmento."""
        return self.start_frame <= frame_index and (
            self.end_frame is None or frame_index < self.end_frame
        )


@dataclass(frozen=True)
class ViewSegmentReport:
    """Resumen seguro de una vista materializado al finalizar el análisis."""

    profile_id: str
    profile_revision: str
    start_frame: int
    end_frame: int
    valid_minutes: int
    discarded_minutes: int
    discard_reason: str | None = None


@dataclass(frozen=True)
class VideoViewPlan:
    """Plan validado que asigna cada frame de un video a una calibración local."""

    profiles: tuple[CameraCalibration, ...]
    segments: tuple[VideoViewSegment, ...]
    schema_version: str = _PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _PLAN_SCHEMA_VERSION:
            raise VideoValidationError("vision.view_plan: unsupported schema version")
        if not self.profiles or not self.segments:
            raise VideoValidationError("vision.view_plan: profiles and segments are required")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise VideoValidationError("vision.view_plan: profile ids must be unique")
        available_profiles = set(profile_ids)
        if self.segments[0].start_frame != 1:
            raise VideoValidationError("vision.view_plan: the first segment must start at frame one")
        for index, segment in enumerate(self.segments):
            if segment.profile_id not in available_profiles:
                raise VideoValidationError("vision.view_plan: segment references an unknown profile")
            if index == len(self.segments) - 1:
                if segment.end_frame is not None:
                    raise VideoValidationError(
                        "vision.view_plan: the final segment must remain open ended"
                    )
                continue
            if segment.end_frame is None or self.segments[index + 1].start_frame != segment.end_frame:
                raise VideoValidationError(
                    "vision.view_plan: segments must be contiguous and non-overlapping"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> VideoViewPlan:
        """Construye un plan desde un payload JSON ya cargado por el consumidor."""
        schema_version = _required_string(payload, "schema_version", "vision.view_plan")
        profiles = tuple(
            _parse_profile(item) for item in _required_list(payload, "profiles", "vision.view_plan")
        )
        segments = tuple(
            _parse_segment(item) for item in _required_list(payload, "segments", "vision.view_plan")
        )
        return cls(profiles=profiles, segments=segments, schema_version=schema_version)

    def resolve(self, frame_index: int, *, width: int, height: int) -> CameraCalibration:
        """Devuelve la calibración aplicable y verifica su resolución antes de usarla."""
        for segment in self.segments:
            if segment.contains(frame_index):
                profile = next(
                    profile for profile in self.profiles if profile.profile_id == segment.profile_id
                )
                profile.validate_frame_size(width, height)
                return profile
        raise VideoValidationError("vision.view_plan: frame is not covered by a segment")

    def segment_index(self, frame_index: int) -> int:
        """Devuelve el índice del segmento que cubre el frame solicitado."""
        for index, segment in enumerate(self.segments):
            if segment.contains(frame_index):
                return index
        raise VideoValidationError("vision.view_plan: frame is not covered by a segment")


def _parse_profile(value: object) -> CameraCalibration:
    payload = _mapping(value, "vision.view_plan.profile")
    frame_size = _required_list(payload, "frame_size", "vision.view_plan.profile")
    if len(frame_size) != 2:
        raise VideoValidationError("vision.view_plan.profile: frame_size must contain width and height")
    references = tuple(
        _parse_reference(item)
        for item in _required_list(payload, "references", "vision.view_plan.profile")
    )
    return CameraCalibration(
        profile_id=_required_string(payload, "profile_id", "vision.view_plan.profile"),
        revision=_required_string(payload, "revision", "vision.view_plan.profile"),
        frame_size=(
            _positive_int(frame_size[0], "vision.view_plan.profile.frame_size"),
            _positive_int(frame_size[1], "vision.view_plan.profile.frame_size"),
        ),
        references=references,
    )


def _parse_reference(value: object) -> CalibrationReference:
    payload = _mapping(value, "vision.view_plan.reference")
    return CalibrationReference(
        reference_id=_required_string(payload, "reference_id", "vision.view_plan.reference"),
        pixel_start=_point(payload.get("pixel_start"), "vision.view_plan.reference.pixel_start"),
        pixel_end=_point(payload.get("pixel_end"), "vision.view_plan.reference.pixel_end"),
        meters=_positive_float(payload.get("meters"), "vision.view_plan.reference.meters"),
    )


def _parse_segment(value: object) -> VideoViewSegment:
    payload = _mapping(value, "vision.view_plan.segment")
    end_frame = payload.get("end_frame")
    return VideoViewSegment(
        start_frame=_positive_int(payload.get("start_frame"), "vision.view_plan.segment.start_frame"),
        end_frame=(
            None
            if end_frame is None
            else _positive_int(end_frame, "vision.view_plan.segment.end_frame")
        ),
        profile_id=_required_string(payload, "profile_id", "vision.view_plan.segment"),
    )


def _mapping(value: object, stage: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise VideoValidationError(f"{stage}: object is required")
    return cast(Mapping[str, object], value)


def _required_list(payload: Mapping[str, object], key: str, stage: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise VideoValidationError(f"{stage}: {key} must be a list")
    return value


def _required_string(payload: Mapping[str, object], key: str, stage: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise VideoValidationError(f"{stage}: {key} must be a non-empty string")
    return value


def _positive_int(value: object, stage: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VideoValidationError(f"{stage}: positive integer is required")
    return value


def _positive_float(value: object, stage: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoValidationError(f"{stage}: finite positive number is required")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise VideoValidationError(f"{stage}: finite positive number is required")
    return parsed


def _point(value: object, stage: str) -> _Point:
    if not isinstance(value, list) or len(value) != 2:
        raise VideoValidationError(f"{stage}: [x, y] is required")
    return (_finite_float(value[0], stage), _finite_float(value[1], stage))


def _finite_float(value: object, stage: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoValidationError(f"{stage}: finite number is required")
    parsed = float(value)
    if not isfinite(parsed):
        raise VideoValidationError(f"{stage}: finite number is required")
    return parsed
