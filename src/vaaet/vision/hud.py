"""Responsive public-facing HUD for annotated traffic videos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np

__all__ = [
    "HudConfig",
    "HudLayout",
    "HudSnapshot",
    "PanelBounds",
    "compute_hud_layout",
    "draw_track_annotation",
    "format_elapsed_time",
    "format_track_label",
    "render_hud",
]


@dataclass(frozen=True)
class HudConfig:
    """User-facing annotation options shared by collection and inference."""

    debug: bool = False


@dataclass(frozen=True)
class HudSnapshot:
    """Frame-level values consumed by the HUD renderer."""

    elapsed_seconds: float
    cumulative_counts: Mapping[str, int]
    average_speed: float | None
    inference_enabled: bool
    state: int | None = None
    confidence: float | None = None
    evidence: float | None = None
    incident_candidate: bool = False
    measurement_quality: float | None = None


@dataclass(frozen=True)
class PanelBounds:
    """Inclusive-exclusive rectangle used by responsive HUD layouts."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class HudLayout:
    """Resolved panel placement for one frame resolution."""

    mode: str
    status: PanelBounds
    counts: PanelBounds | None


_VEHICLE_LABELS: Mapping[str, str] = {
    "car": "AUTO",
    "truck": "CAMION",
    "bus": "BUS",
    "motorcycle": "MOTO",
    "bicycle": "BICI",
}

_VEHICLE_COLORS: Mapping[str, tuple[int, int, int]] = {
    "car": (80, 210, 80),
    "truck": (40, 150, 255),
    "bus": (80, 190, 255),
    "motorcycle": (255, 170, 30),
    "bicycle": (230, 130, 230),
}

_STATE_COLORS: Mapping[int, tuple[int, int, int]] = {
    0: (92, 173, 46),
    1: (52, 177, 242),
    2: (65, 111, 231),
    3: (47, 47, 211),
}

_STATE_TITLES: Mapping[int, str] = {
    0: "TRAFICO NORMAL",
    1: "TRAFICO REDUCIDO",
    2: "CONGESTION",
    3: "ACCIDENTE CONFIRMADO",
}

_NEUTRAL_COLOR = (224, 144, 74)
_TEXT_COLOR = (245, 247, 250)
_MUTED_TEXT_COLOR = (191, 199, 209)
_CARD_TINT = (24, 27, 32)
_INCIDENT_COLOR = (48, 48, 225)


def format_elapsed_time(elapsed_seconds: float) -> str:
    """Return a stable clock label suitable for public video overlays."""
    total_seconds = max(int(elapsed_seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_speed(average_speed: float | None, *, debug: bool = False) -> str:
    if average_speed is None:
        return "-- km/h"
    return f"{average_speed:.1f} km/h" if debug else f"{average_speed:.0f} km/h"


def _total_vehicle_count(counts: Mapping[str, int]) -> int:
    return sum(max(int(value), 0) for value in counts.values())


def _format_total_vehicle_count(counts: Mapping[str, int]) -> str:
    return f"TOTAL VEHICULOS: {_total_vehicle_count(counts)}"


def _vehicle_count_labels(counts: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(
        f"{_VEHICLE_LABELS[kind]} {max(int(counts.get(kind, 0)), 0)}"
        for kind in ("car", "truck", "bus", "motorcycle", "bicycle")
    )


def format_track_label(
    vehicle_type: str,
    *,
    speed: float | None,
    stationary: bool,
    track_id: int,
    debug: bool,
) -> str:
    """Build a concise Spanish vehicle label with optional technical identity."""
    vehicle = _VEHICLE_LABELS.get(vehicle_type, vehicle_type.upper())
    if stationary:
        motion = "DETENIDO"
    elif speed is None:
        motion = "SIN VELOCIDAD"
    elif debug:
        motion = f"{speed:.1f} km/h"
    else:
        motion = f"{speed:.0f} km/h"
    identity = f" #{track_id}" if debug else ""
    return f"{vehicle}{identity} | {motion}"


def compute_hud_layout(
    frame_width: int,
    frame_height: int,
    *,
    debug: bool = False,
) -> HudLayout:
    """Resolve two-corner cards with a safe compact fallback."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    margin = max(2, round(min(frame_width, frame_height) * 0.018))
    if frame_width < 640:
        panel_height = min(
            frame_height - 2 * margin,
            max(42, round(frame_height * (0.43 if debug else 0.36))),
        )
        return HudLayout(
            mode="compact",
            status=PanelBounds(margin, margin, frame_width - margin, margin + panel_height),
            counts=None,
        )

    scale = max(0.55, min(1.4, min(frame_width / 1280, frame_height / 720)))
    panel_height = round((168 if debug else 146) * scale)
    panel_height = min(panel_height, frame_height - 2 * margin)
    if frame_width >= 960:
        status_width = min(round(430 * scale), round(frame_width * 0.42))
        counts_width = min(round(410 * scale), round(frame_width * 0.38))
        mode = "wide"
    else:
        available = frame_width - 3 * margin
        status_width = available // 2
        counts_width = available - status_width
        mode = "medium"
    return HudLayout(
        mode=mode,
        status=PanelBounds(margin, margin, margin + status_width, margin + panel_height),
        counts=PanelBounds(
            frame_width - margin - counts_width,
            margin,
            frame_width - margin,
            margin + panel_height,
        ),
    )


def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(0, min(radius, height // 2, width // 2))
    if radius == 0:
        mask[:, :] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    for center in (
        (radius, radius),
        (width - radius, radius),
        (radius, height - radius),
        (width - radius, height - radius),
    ):
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask


def _draw_blurred_card(
    frame: np.ndarray,
    bounds: PanelBounds,
    *,
    accent: tuple[int, int, int] | None = None,
) -> None:
    import cv2

    x1 = max(0, min(bounds.x1, frame.shape[1] - 1))
    y1 = max(0, min(bounds.y1, frame.shape[0] - 1))
    x2 = max(x1 + 1, min(bounds.x2, frame.shape[1]))
    y2 = max(y1 + 1, min(bounds.y2, frame.shape[0]))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    kernel = min(15, roi.shape[0], roi.shape[1])
    if kernel % 2 == 0:
        kernel -= 1
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0) if kernel >= 3 else roi.copy()
    tint = np.full_like(blurred, _CARD_TINT)
    composite = cv2.addWeighted(blurred, 0.68, tint, 0.32, 0)
    radius = max(3, round(min(roi.shape[:2]) * 0.09))
    mask = _rounded_mask(roi.shape[0], roi.shape[1], radius)
    np.copyto(roi, composite, where=mask[..., None].astype(bool))
    if accent is not None:
        accent_width = max(3, round(bounds.width * 0.014))
        cv2.rectangle(
            frame,
            (x1, y1 + radius),
            (min(x1 + accent_width, x2 - 1), y2 - radius),
            accent,
            -1,
            cv2.LINE_AA,
        )


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = _TEXT_COLOR,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (5, 7, 10),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_state_icon(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: int,
    state: int | None,
    color: tuple[int, int, int],
) -> None:
    import cv2

    x, y = origin
    radius = max(5, size // 2)
    cv2.circle(frame, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
    white = (250, 250, 250)
    thickness = max(1, size // 8)
    if state == 0:
        cv2.line(frame, (x + size // 4, y + size // 2), (x + size // 2, y + 3 * size // 4), white, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + size // 2, y + 3 * size // 4), (x + 4 * size // 5, y + size // 4), white, thickness, cv2.LINE_AA)
    elif state in {1, 2}:
        bars = 2 if state == 1 else 3
        for index in range(bars):
            y_line = y + size // 3 + index * max(3, size // 5)
            cv2.line(frame, (x + size // 4, y_line), (x + 3 * size // 4, y_line), white, thickness, cv2.LINE_AA)
    elif state == 3:
        _put_text(frame, "!", (x + size // 3, y + 4 * size // 5), scale=max(0.35, size / 32), thickness=2)
    else:
        cv2.circle(frame, (x + radius, y + radius), max(1, size // 7), white, -1, cv2.LINE_AA)


def _draw_speedometer(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    import cv2

    x, y = origin
    center = (x + size // 2, y + size // 2)
    radius = max(4, size // 2 - 1)
    cv2.ellipse(frame, center, (radius, radius), 0, 190, 350, color, 2, cv2.LINE_AA)
    angle = 315 * pi / 180
    tip = (
        round(center[0] + radius * 0.65 * cos(angle)),
        round(center[1] + radius * 0.65 * sin(angle)),
    )
    cv2.line(frame, center, tip, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, size // 12), color, -1, cv2.LINE_AA)


def _status_values(snapshot: HudSnapshot) -> tuple[str, str, tuple[int, int, int]]:
    if not snapshot.inference_enabled:
        return "RECOLECCION DE TELEMETRIA", "MODO ADQUISICION", _NEUTRAL_COLOR
    if snapshot.state is None:
        return "ANALIZANDO", "ESTADO AL COMPLETAR 1 MIN", _NEUTRAL_COLOR
    return (
        _STATE_TITLES.get(snapshot.state, "ESTADO DESCONOCIDO"),
        "ESTADO ESTABLE",
        _STATE_COLORS.get(snapshot.state, _NEUTRAL_COLOR),
    )


def _render_status_card(
    frame: np.ndarray,
    bounds: PanelBounds,
    snapshot: HudSnapshot,
    config: HudConfig,
) -> None:
    title, subtitle, state_color = _status_values(snapshot)
    _draw_blurred_card(frame, bounds, accent=state_color)
    scale = max(0.42, min(1.35, bounds.height / (168 if config.debug else 146)))
    icon_size = max(15, round(27 * scale))
    left = bounds.x1 + max(10, round(18 * scale))
    top = bounds.y1 + max(9, round(13 * scale))
    _draw_state_icon(frame, (left, top), icon_size, snapshot.state, state_color)
    text_left = left + icon_size + max(7, round(10 * scale))
    _put_text(frame, title, (text_left, top + icon_size // 2), scale=0.58 * scale, thickness=2)
    _put_text(frame, subtitle, (text_left, top + icon_size), scale=0.32 * scale, color=_MUTED_TEXT_COLOR)

    speed_top = bounds.y1 + round(bounds.height * 0.48)
    speed_icon_size = max(14, round(23 * scale))
    _draw_speedometer(frame, (left, speed_top - speed_icon_size + 3), speed_icon_size, state_color)
    speed_label = _format_speed(snapshot.average_speed, debug=config.debug)
    _put_text(frame, "VELOCIDAD MEDIA", (text_left, speed_top - round(8 * scale)), scale=0.31 * scale, color=_MUTED_TEXT_COLOR)
    _put_text(frame, speed_label, (text_left, speed_top + round(14 * scale)), scale=0.52 * scale, thickness=2)
    time_text = f"TIEMPO {format_elapsed_time(snapshot.elapsed_seconds)}"
    time_offset = 34 if config.debug and not snapshot.incident_candidate else 13
    _put_text(
        frame,
        time_text,
        (left, bounds.y2 - max(9, round(time_offset * scale))),
        scale=0.34 * scale,
        color=_MUTED_TEXT_COLOR,
    )

    if snapshot.incident_candidate:
        banner_y = bounds.y2 - max(31, round(34 * scale))
        _put_text(frame, "POSIBLE INCIDENTE - REVISAR", (text_left, banner_y), scale=0.38 * scale, color=_INCIDENT_COLOR, thickness=2)
    elif config.debug:
        confidence = "--" if snapshot.confidence is None else f"{snapshot.confidence:.0%}"
        evidence = "--" if snapshot.evidence is None else f"{snapshot.evidence:.2f}"
        quality = "--" if snapshot.measurement_quality is None else f"{snapshot.measurement_quality:.0%}"
        technical = f"CONF {confidence} | EVID {evidence} | CALIDAD {quality}"
        _put_text(frame, technical, (left, bounds.y2 - max(9, round(13 * scale))), scale=0.27 * scale, color=_MUTED_TEXT_COLOR)


def _render_counts_card(
    frame: np.ndarray,
    bounds: PanelBounds,
    snapshot: HudSnapshot,
) -> None:
    _draw_blurred_card(frame, bounds)
    scale = max(0.42, min(1.35, bounds.height / 146))
    left = bounds.x1 + max(10, round(16 * scale))
    title_y = bounds.y1 + max(21, round(30 * scale))
    _put_text(
        frame,
        _format_total_vehicle_count(snapshot.cumulative_counts),
        (left, title_y),
        scale=0.43 * scale,
        color=_MUTED_TEXT_COLOR,
        thickness=2,
    )

    columns = 3
    cell_width = max(1, (bounds.width - 2 * (left - bounds.x1)) // columns)
    row_height = max(24, round(46 * scale))
    first_row_y = bounds.y1 + round(66 * scale)
    for index, label in enumerate(_vehicle_count_labels(snapshot.cumulative_counts)):
        row, column = divmod(index, columns)
        x = left + column * cell_width
        y = first_row_y + row * row_height
        _put_text(frame, label, (x, y), scale=0.34 * scale, thickness=1)


def _render_compact(
    frame: np.ndarray,
    bounds: PanelBounds,
    snapshot: HudSnapshot,
    config: HudConfig,
) -> None:
    title, _, state_color = _status_values(snapshot)
    _draw_blurred_card(frame, bounds, accent=state_color)
    scale = max(0.24, min(0.58, bounds.width / 640))
    left = bounds.x1 + max(7, round(14 * scale))
    top = bounds.y1 + max(12, round(23 * scale))
    icon_size = max(10, round(22 * scale))
    _draw_state_icon(frame, (left, top - icon_size + 2), icon_size, snapshot.state, state_color)
    _put_text(frame, title, (left + icon_size + 5, top), scale=0.48 * scale, thickness=2)
    speed = _format_speed(snapshot.average_speed, debug=config.debug)
    total = _format_total_vehicle_count(snapshot.cumulative_counts)
    summary = f"{speed} | {total} | {format_elapsed_time(snapshot.elapsed_seconds)}"
    _put_text(frame, summary, (left, min(bounds.y2 - 5, top + max(14, round(25 * scale)))), scale=0.39 * scale)
    if bounds.width >= 320:
        y = min(bounds.y2 - 4, top + max(28, round(48 * scale)))
        labels = _vehicle_count_labels(snapshot.cumulative_counts)
        cell = max(1, (bounds.width - 2 * (left - bounds.x1)) // len(labels))
        for index, label in enumerate(labels):
            x = left + index * cell
            _put_text(frame, label, (x, y), scale=0.31 * scale)
    if snapshot.incident_candidate:
        _put_text(frame, "POSIBLE INCIDENTE - REVISAR", (left, bounds.y2 - 5), scale=0.36 * scale, color=_INCIDENT_COLOR, thickness=2)
    elif config.debug:
        confidence = "--" if snapshot.confidence is None else f"{snapshot.confidence:.0%}"
        _put_text(frame, f"DEBUG CONF {confidence}", (left, bounds.y2 - 5), scale=0.28 * scale, color=_MUTED_TEXT_COLOR)


def render_hud(
    frame: np.ndarray,
    snapshot: HudSnapshot,
    config: HudConfig | None = None,
) -> HudLayout:
    """Render the public HUD in place and return its resolved layout."""
    active_config = config or HudConfig()
    height, width = frame.shape[:2]
    layout = compute_hud_layout(width, height, debug=active_config.debug)
    if layout.mode == "compact":
        _render_compact(frame, layout.status, snapshot, active_config)
    else:
        _render_status_card(frame, layout.status, snapshot, active_config)
        assert layout.counts is not None
        _render_counts_card(frame, layout.counts, snapshot)
    return layout


def draw_track_annotation(
    frame: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    vehicle_type: str,
    track_id: int,
    speed: float | None,
    stationary: bool,
    config: HudConfig | None = None,
) -> None:
    """Draw a vehicle box and a concise public or technical label."""
    import cv2

    active_config = config or HudConfig()
    color = _VEHICLE_COLORS.get(vehicle_type, (220, 220, 220))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = format_track_label(
        vehicle_type,
        speed=speed,
        stationary=stationary,
        track_id=track_id,
        debug=active_config.debug,
    )
    _put_text(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        scale=0.48,
        color=color,
        thickness=1,
    )
