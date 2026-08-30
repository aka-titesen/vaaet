# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptador de laboratorio para cargar planes privados de vistas de video."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from vaaet.exceptions import VideoValidationError
from vaaet.vision.view_plan import VideoViewPlan

from vaaet_ml.exceptions import RuntimeConfigurationError

__all__ = ["load_video_view_plan"]


def load_video_view_plan(path: str | Path | None) -> VideoViewPlan | None:
    """Carga un JSON local opcional sin revelar su ubicación ni contenido al fallar."""
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise RuntimeConfigurationError("The configured view plan is unavailable.")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeConfigurationError("The configured view plan cannot be read.") from error
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise RuntimeConfigurationError("The configured view plan must contain a JSON object.")
    try:
        return VideoViewPlan.from_mapping(cast(Mapping[str, object], payload))
    except VideoValidationError as error:
        raise RuntimeConfigurationError("The configured view plan is invalid.") from error
