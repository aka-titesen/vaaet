# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Pruebas del adaptador local para planes privados de vistas."""

from __future__ import annotations

import json

import pytest

from vaaet_ml.exceptions import RuntimeConfigurationError
from vaaet_ml.view_plan import load_video_view_plan


def _payload() -> dict[str, object]:
    return {
        "schema_version": "vaaet-view-plan-v1",
        "profiles": [
            {
                "profile_id": "synthetic-cam",
                "revision": "v1",
                "frame_size": [64, 48],
                "references": [
                    {
                        "reference_id": "far",
                        "pixel_start": [0, 10],
                        "pixel_end": [10, 10],
                        "meters": 1,
                    },
                    {
                        "reference_id": "near",
                        "pixel_start": [0, 40],
                        "pixel_end": [20, 40],
                        "meters": 1,
                    },
                ],
            }
        ],
        "segments": [{"start_frame": 1, "end_frame": None, "profile_id": "synthetic-cam"}],
    }


def test_load_video_view_plan_returns_none_when_opted_out() -> None:
    assert load_video_view_plan(None) is None


def test_load_video_view_plan_reads_valid_local_json(tmp_path) -> None:
    path = tmp_path / "view-plan.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    plan = load_video_view_plan(path)

    assert plan is not None
    assert plan.profiles[0].profile_id == "synthetic-cam"


def test_load_video_view_plan_redacts_private_path_on_failure(tmp_path) -> None:
    private_path = tmp_path / "private-camera-plan.json"

    with pytest.raises(RuntimeConfigurationError, match="unavailable") as error:
        load_video_view_plan(private_path)

    assert str(private_path) not in str(error.value)


def test_load_video_view_plan_chains_invalid_contract_without_payload(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match="invalid") as error:
        load_video_view_plan(path)

    assert error.value.__cause__ is not None
