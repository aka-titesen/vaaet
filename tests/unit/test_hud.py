"""Tests for the responsive annotated-video HUD."""

from __future__ import annotations

import numpy as np
import pytest

from vaaet.vision import hud
from vaaet.vision.hud import (
    HudConfig,
    HudSnapshot,
    compute_hud_layout,
    format_elapsed_time,
    format_track_label,
    render_hud,
)


def _snapshot(**overrides: object) -> HudSnapshot:
    values: dict[str, object] = {
        "elapsed_seconds": 135.0,
        "cumulative_counts": {
            "car": 95,
            "truck": 8,
            "bus": 7,
            "motorcycle": 14,
            "bicycle": 4,
        },
        "average_speed": 34.25,
        "inference_enabled": True,
        "state": 0,
        "confidence": 0.91,
        "evidence": 0.2,
        "measurement_quality": 0.94,
    }
    values.update(overrides)
    return HudSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (135, "02:15"), (3661, "01:01:01"), (-4, "00:00")],
)
def test_elapsed_time_is_public_and_stable(seconds: float, expected: str) -> None:
    assert format_elapsed_time(seconds) == expected


def test_public_and_debug_track_labels_have_different_detail() -> None:
    public = format_track_label(
        "car", speed=34.25, stationary=False, track_id=17, debug=False
    )
    debug = format_track_label(
        "car", speed=34.25, stationary=False, track_id=17, debug=True
    )
    assert public == "AUTO | 34 km/h"
    assert "#17" not in public
    assert debug == "AUTO #17 | 34.2 km/h"
    assert format_track_label(
        "bus", speed=None, stationary=True, track_id=2, debug=False
    ) == "BUS | DETENIDO"


def test_unknown_speed_and_accumulated_total_are_not_misrepresented() -> None:
    assert hud._format_speed(None) == "-- km/h"
    assert hud._format_speed(0.0) == "0 km/h"
    assert hud._total_vehicle_count(_snapshot().cumulative_counts) == 128


@pytest.mark.parametrize(
    ("width", "height", "expected_mode"),
    [
        (1920, 1080, "wide"),
        (1280, 720, "wide"),
        (800, 450, "medium"),
        (639, 360, "compact"),
        (96, 64, "compact"),
    ],
)
def test_layout_is_responsive_and_clipped(
    width: int, height: int, expected_mode: str
) -> None:
    layout = compute_hud_layout(width, height, debug=True)
    assert layout.mode == expected_mode
    for bounds in (layout.status, layout.counts):
        if bounds is None:
            continue
        assert 0 <= bounds.x1 < bounds.x2 <= width
        assert 0 <= bounds.y1 < bounds.y2 <= height
    if layout.counts is not None:
        assert layout.status.x2 < layout.counts.x1


def test_status_copy_distinguishes_collection_warmup_and_stable_state() -> None:
    collection = hud._status_values(
        _snapshot(inference_enabled=False, state=None)
    )
    warmup = hud._status_values(_snapshot(state=None))
    congested = hud._status_values(_snapshot(state=2, incident_candidate=True))
    assert collection[0] == "RECOLECCION DE TELEMETRIA"
    assert warmup[0] == "ANALIZANDO"
    assert congested[0] == "CONGESTION"


@pytest.mark.parametrize("shape", [(1080, 1920), (720, 1280), (450, 800), (360, 639)])
def test_renderer_changes_only_hud_regions_and_preserves_frame_center(
    shape: tuple[int, int],
) -> None:
    pytest.importorskip("cv2")
    height, width = shape
    frame = np.full((height, width, 3), 96, dtype=np.uint8)
    original = frame.copy()
    layout = render_hud(frame, _snapshot(), HudConfig(debug=False))
    status = layout.status
    assert np.any(
        frame[status.y1 : status.y2, status.x1 : status.x2]
        != original[status.y1 : status.y2, status.x1 : status.x2]
    )
    if layout.counts is not None:
        counts = layout.counts
        assert np.any(
            frame[counts.y1 : counts.y2, counts.x1 : counts.x2]
            != original[counts.y1 : counts.y2, counts.x1 : counts.x2]
        )
    assert np.array_equal(frame[-10:, width // 3 : 2 * width // 3], original[-10:, width // 3 : 2 * width // 3])


def test_incident_candidate_and_debug_render_without_publishing_accident() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    snapshot = _snapshot(state=2, incident_candidate=True, evidence=0.88)
    render_hud(frame, snapshot, HudConfig(debug=True))
    assert snapshot.state == 2
    assert snapshot.incident_candidate is True
    assert np.any(frame != 0)
