# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from vaaet_ml.exceptions import RuntimeConfigurationError
from vaaet_ml.notebook_io import resolve_video_input


def _write_video(path: Path, content: bytes = b"mp4") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_explicit_video_path_has_priority_over_colab_upload(tmp_path: Path) -> None:
    explicit = _write_video(tmp_path / "already-uploaded.mp4")

    def uploader(**_kwargs: object) -> Mapping[str, bytes]:
        raise AssertionError("No debe abrirse el selector para una ruta explícita.")

    selected = resolve_video_input(
        explicit,
        in_colab=True,
        uploader=uploader,
        staging_directory=tmp_path / "content",
        local_fallback=None,
    )

    assert selected == explicit.resolve()


def test_colab_upload_uses_explicit_staging_directory(tmp_path: Path) -> None:
    staging = tmp_path / "content"
    uploaded_video = _write_video(staging / "traffic.mp4")
    observed_target: list[str] = []

    def uploader(*, target_dir: str) -> Mapping[str, bytes]:
        observed_target.append(target_dir)
        return {str(uploaded_video): b"payload local al adaptador"}

    selected = resolve_video_input(
        None,
        in_colab=True,
        uploader=uploader,
        staging_directory=staging,
        local_fallback=tmp_path / "unused.mp4",
    )

    assert selected == uploaded_video.resolve()
    assert observed_target == [str(staging.resolve())]


def test_local_execution_uses_only_declared_fallback(tmp_path: Path) -> None:
    fallback = _write_video(tmp_path / "sample.mp4")

    selected = resolve_video_input(
        None,
        in_colab=False,
        uploader=None,
        staging_directory=tmp_path / "content",
        local_fallback=fallback,
    )

    assert selected == fallback.resolve()


@pytest.mark.parametrize(
    ("uploaded", "message"),
    [
        ({}, "No se seleccionó ningún video"),
        ({"one.mp4": b"1", "two.mp4": b"2"}, "un solo archivo MP4"),
    ],
)
def test_colab_rejects_cancelled_or_multiple_uploads(
    tmp_path: Path, uploaded: Mapping[str, bytes], message: str
) -> None:
    with pytest.raises(RuntimeConfigurationError, match=message):
        resolve_video_input(
            None,
            in_colab=True,
            uploader=lambda **_kwargs: uploaded,
            staging_directory=tmp_path,
            local_fallback=None,
        )


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    wrong_format = _write_video(tmp_path / "traffic.avi")

    with pytest.raises(RuntimeConfigurationError, match="extensión .mp4"):
        resolve_video_input(
            wrong_format,
            in_colab=False,
            uploader=None,
            staging_directory=tmp_path,
            local_fallback=None,
        )


def test_rejects_empty_video(tmp_path: Path) -> None:
    empty = _write_video(tmp_path / "empty.mp4", b"")

    with pytest.raises(RuntimeConfigurationError, match="está vacío"):
        resolve_video_input(
            empty,
            in_colab=False,
            uploader=None,
            staging_directory=tmp_path,
            local_fallback=None,
        )


def test_colab_rejects_upload_outside_staging_directory(tmp_path: Path) -> None:
    staging = tmp_path / "content"
    outside = _write_video(tmp_path / "outside.mp4")

    with pytest.raises(RuntimeConfigurationError, match="fuera del directorio"):
        resolve_video_input(
            None,
            in_colab=True,
            uploader=lambda **_kwargs: {str(outside): b"payload"},
            staging_directory=staging,
            local_fallback=None,
        )
