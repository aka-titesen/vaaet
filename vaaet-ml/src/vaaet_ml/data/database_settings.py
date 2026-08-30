# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Configuración segura y perfiles de PostgreSQL para el laboratorio."""

from __future__ import annotations

import os
import stat
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from vaaet.logging import get_logger

from vaaet_ml.exceptions import DatabaseNotConfiguredError
from vaaet_ml.settings import DEFAULT_DB_PORT

logger = get_logger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


class DatabaseProfile(str, Enum):
    """Identidad de mínimo privilegio usada por cada workflow."""

    COLLECTION = "collection"
    INFERENCE = "inference"
    TRAINING = "training"
    REVIEW = "review"


_PROFILE_ENV_PREFIX = {
    DatabaseProfile.COLLECTION: "VAAET_COLLECTION_DB",
    DatabaseProfile.INFERENCE: "VAAET_INFERENCE_DB",
    DatabaseProfile.TRAINING: "VAAET_TRAINING_DB",
    DatabaseProfile.REVIEW: "VAAET_REVIEW_DB",
}


@dataclass(frozen=True, repr=False)
class DatabaseSettings:
    """Configuración validada cuya representación nunca expone secretos."""

    profile: DatabaseProfile
    host: str
    port: int
    database: str
    username: str
    password: str = field(repr=False)
    sslmode: str = "verify-full"
    sslrootcert: str | None = None
    connect_timeout_seconds: int = 10
    application_name: str | None = None
    _temporary_root_cert: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.host or not self.database or not self.username or not self.password:
            raise DatabaseNotConfiguredError(
                f"Incomplete PostgreSQL configuration for profile={self.profile.value}."
            )
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("PostgreSQL port must be between 1 and 65535.")
        if self.sslmode not in _SSL_MODES:
            raise ValueError(f"Unsupported PostgreSQL sslmode: {self.sslmode}")
        if self.sslmode == "disable" and self.host.lower() not in _LOCAL_HOSTS:
            raise ValueError("sslmode=disable is allowed only for an explicit localhost endpoint.")
        if self.sslmode == "verify-full" and not self.sslrootcert:
            raise DatabaseNotConfiguredError(
                "sslmode=verify-full requires VAAET_DB_SSLROOTCERT or "
                "VAAET_DB_SSLROOTCERT_PEM. Use sslmode=require only as an explicit, "
                "documented fallback when the provider cannot expose a CA certificate."
            )

    def __repr__(self) -> str:
        return (
            "DatabaseSettings("
            f"profile={self.profile.value!r}, host={self.host!r}, port={self.port!r}, "
            f"database={self.database!r}, username='<redacted>', password='<redacted>', "
            f"sslmode={self.sslmode!r})"
        )

    @property
    def application(self) -> str:
        """Devuelve una identidad segura para observabilidad del cliente SQL."""

        return self.application_name or f"vaaet-{self.profile.value}-4.5.3"


def _colab_secret(name: str) -> str | None:
    """Lee un secreto de Colab sin hacer de su ausencia un error de runtime."""

    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:  # pragma: no cover - frontera de Colab no determinista
        return None
    return str(value).strip() if value else None


def _setting(name: str) -> str | None:
    """Obtiene secretos de Colab antes de consultar variables de entorno."""

    return _colab_secret(name) or (os.environ.get(name) or "").strip() or None


def _materialize_root_certificate(pem: str) -> str:
    """Materializa un PEM temporal con permisos exclusivos del proceso actual."""

    descriptor, path = tempfile.mkstemp(prefix="vaaet-postgres-ca-", suffix=".pem")
    try:
        os.write(descriptor, pem.encode("utf-8"))
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def load_database_settings(
    profile: DatabaseProfile | str,
    *,
    env_file: str | Path | None = None,
    allow_legacy: bool = True,
) -> DatabaseSettings:
    """Carga un perfil desde secretos Colab o entorno sin registrar credenciales."""

    active_profile = DatabaseProfile(profile)
    if env_file is not None:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise DatabaseNotConfiguredError(
                "python-dotenv is required when env_file is supplied."
            ) from exc
        load_dotenv(dotenv_path=Path(env_file), override=False)

    prefix = _PROFILE_ENV_PREFIX[active_profile]
    host = _setting("VAAET_DB_HOST")
    port = _setting("VAAET_DB_PORT") or DEFAULT_DB_PORT
    database = _setting("VAAET_DB_NAME")
    username = _setting(f"{prefix}_USER")
    password = _setting(f"{prefix}_PASSWORD")

    if allow_legacy and not all((host, database, username, password)):
        legacy = {
            "host": _setting("DB_HOST"),
            "port": _setting("DB_PORT") or DEFAULT_DB_PORT,
            "database": _setting("DB_NAME"),
            "username": _setting("DB_USER"),
            "password": _setting("DB_PASSWORD"),
        }
        if all((legacy["host"], legacy["database"], legacy["username"], legacy["password"])):
            warnings.warn(
                "DB_* variables are deprecated and will be removed in VAAET 5.0; "
                "use VAAET_DB_* plus profile-specific credentials.",
                FutureWarning,
                stacklevel=2,
            )
            host = host or legacy["host"]
            port = port or legacy["port"]
            database = database or legacy["database"]
            username = username or legacy["username"]
            password = password or legacy["password"]

    missing = [
        name
        for name, value in {
            "VAAET_DB_HOST": host,
            "VAAET_DB_NAME": database,
            f"{prefix}_USER": username,
            f"{prefix}_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise DatabaseNotConfiguredError(
            f"PostgreSQL profile={active_profile.value} is not configured; missing: "
            + ", ".join(missing)
        )

    sslmode = (_setting("VAAET_DB_SSLMODE") or "verify-full").lower()
    sslrootcert = _setting("VAAET_DB_SSLROOTCERT")
    temporary_cert = False
    pem = _setting("VAAET_DB_SSLROOTCERT_PEM")
    if pem and not sslrootcert:
        sslrootcert = _materialize_root_certificate(pem.replace("\\n", "\n"))
        temporary_cert = True
    if sslmode == "require":
        logger.warning(
            "PostgreSQL TLS encrypts transport but does not verify server identity (sslmode=require)."
        )

    return DatabaseSettings(
        profile=active_profile,
        host=str(host),
        port=int(port),
        database=str(database),
        username=str(username),
        password=str(password),
        sslmode=sslmode,
        sslrootcert=sslrootcert,
        connect_timeout_seconds=int(_setting("VAAET_DB_CONNECT_TIMEOUT") or "10"),
        _temporary_root_cert=temporary_cert,
    )


def get_optional_database_settings(
    profile: DatabaseProfile | str,
    *,
    env_file: str | Path | None = None,
) -> DatabaseSettings | None:
    """Devuelve un perfil opcional sin ocultar errores de configuración válidos."""

    try:
        return load_database_settings(profile, env_file=env_file)
    except DatabaseNotConfiguredError:
        logger.info("Optional PostgreSQL profile=%s is not configured", DatabaseProfile(profile).value)
        return None


def load_reviewer_id() -> str:
    """Carga el seudónimo estable del revisor sin registrarlo en logs."""

    reviewer_id = _setting("VAAET_REVIEWER_ID")
    if not reviewer_id:
        raise DatabaseNotConfiguredError(
            "VAAET_REVIEWER_ID is required in Colab Secrets or the local environment."
        )
    return reviewer_id


__all__ = [
    "DatabaseProfile",
    "DatabaseSettings",
    "get_optional_database_settings",
    "load_database_settings",
    "load_reviewer_id",
]
