# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Colab-safe PostgreSQL backup reader preparation for training ingestion."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from vaaet.logging import get_logger

logger = get_logger(__name__)

def resolve_pg_restore_for_backup(
    backup_path: Path,
    csv_path: Path,
    *,
    in_colab: bool,
) -> str | None:
    """Return a compatible ``pg_restore`` only when the explicit backup needs it."""

    pg_restore_path = shutil.which("pg_restore")
    if not backup_path.is_file() or csv_path.is_file():
        return pg_restore_path
    if in_colab:
        pg_restore_path = str(_install_colab_pg_restore_17())
    if pg_restore_path is None:
        raise RuntimeError(
            "No pg_restore executable is available. Install PostgreSQL 17 or upload "
            "traffic_data_raw.csv instead."
        )
    version = subprocess.check_output([pg_restore_path, "--version"], text=True).strip()
    logger.info("Lector de backup PostgreSQL disponible: version=%s", version)
    return pg_restore_path


def _install_colab_pg_restore_17() -> Path:
    binary = Path("/usr/lib/postgresql/17/bin/pg_restore")
    if not binary.is_file():
        logger.info("Preparando el cliente PostgreSQL 17 para leer el backup.")
        environment = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        pgdg_directory = Path("/usr/share/postgresql-common/pgdg")
        pgdg_key = pgdg_directory / "apt.postgresql.org.asc"
        pgdg_source = Path("/etc/apt/sources.list.d/pgdg.sources")
        try:
            subprocess.check_call(["apt-get", "update", "-qq"], env=environment)
            subprocess.check_call(
                ["apt-get", "install", "-y", "-qq", "curl", "ca-certificates"],
                env=environment,
            )
            subprocess.check_call(["install", "-d", str(pgdg_directory)])
            subprocess.check_call(
                [
                    "curl", "--fail", "--silent", "--show-error", "--retry", "3",
                    "--output", str(pgdg_key),
                    "https://www.postgresql.org/media/keys/ACCC4CF8.asc",
                ]
            )
            os_release = dict(
                line.split("=", 1)
                for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            codename = os_release.get("VERSION_CODENAME", "").strip().strip('"')
            if not codename:
                raise RuntimeError("Could not determine the Ubuntu release codename")
            architecture = subprocess.check_output(
                ["dpkg", "--print-architecture"], text=True
            ).strip()
            pgdg_source.write_text(
                "Types: deb\n"
                "URIs: https://apt.postgresql.org/pub/repos/apt\n"
                f"Suites: {codename}-pgdg\n"
                f"Architectures: {architecture}\n"
                "Components: main\n"
                f"Signed-By: {pgdg_key}\n",
                encoding="utf-8",
            )
            subprocess.check_call(["apt-get", "update", "-qq"], env=environment)
            subprocess.check_call(
                ["apt-get", "install", "-y", "-qq", "postgresql-client-17"],
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Could not install PostgreSQL 17 from the official PGDG repository. "
                "Retry after reconnecting or upload traffic_data_raw.csv instead."
            ) from exc
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(
            "PostgreSQL 17 installation did not provide an executable. "
            "Upload traffic_data_raw.csv instead."
        )
    return binary
