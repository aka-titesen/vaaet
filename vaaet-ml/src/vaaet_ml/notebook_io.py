# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptadores explícitos de entrada y salida para notebooks VAAET."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from vaaet_ml.exceptions import RuntimeConfigurationError

UploadPayload = Mapping[str, bytes]
VideoUploader = Callable[..., UploadPayload]


def _validate_mp4(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.casefold() != ".mp4":
        raise RuntimeConfigurationError("Seleccioná exactamente un archivo con extensión .mp4.")
    if not resolved.is_file():
        raise RuntimeConfigurationError("El archivo MP4 seleccionado no existe.")
    if resolved.stat().st_size <= 0:
        raise RuntimeConfigurationError("El archivo MP4 seleccionado está vacío.")
    return resolved


def resolve_video_input(
    explicit_path: str | Path | None,
    *,
    in_colab: bool,
    uploader: VideoUploader | None,
    staging_directory: str | Path,
    local_fallback: str | Path | None,
) -> Path:
    """Resuelve un único MP4 sin confundir carga, descarga o persistencia."""

    if explicit_path is not None:
        return _validate_mp4(Path(explicit_path))

    if not in_colab:
        if local_fallback is None:
            raise RuntimeConfigurationError(
                "Definí una ruta explícita o un video local de ejemplo antes de continuar."
            )
        return _validate_mp4(Path(local_fallback))

    if uploader is None:
        raise RuntimeConfigurationError("El selector de archivos de Colab no está disponible.")

    staging_root = Path(staging_directory).expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    uploaded = uploader(target_dir=str(staging_root))
    uploaded_names = tuple(uploaded)
    if not uploaded_names:
        raise RuntimeConfigurationError("No se seleccionó ningún video.")
    if len(uploaded_names) != 1:
        raise RuntimeConfigurationError("Seleccioná un solo archivo MP4 por ejecución.")

    uploaded_path = Path(uploaded_names[0]).expanduser().resolve()
    if uploaded_path.parent != staging_root:
        raise RuntimeConfigurationError("Colab devolvió una ruta fuera del directorio de trabajo.")
    return _validate_mp4(uploaded_path)


__all__ = ["resolve_video_input"]
