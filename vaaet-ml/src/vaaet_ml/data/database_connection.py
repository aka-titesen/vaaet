# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Conexión, reintentos y diagnóstico seguro de PostgreSQL."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool
from vaaet.logging import get_logger

from vaaet_ml.data.database_settings import (
    DatabaseProfile,
    DatabaseSettings,
    load_database_settings,
)
from vaaet_ml.exceptions import DatabaseOperationError
from vaaet_ml.settings import DATABASE_SCHEMAS, DEFAULT_DB_PORT

logger = get_logger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class DatabaseHealth:
    """Diagnóstico no secreto que puede mostrarse en una salida de notebook."""

    profile: str
    host: str
    port: int
    database: str
    server_version: str
    current_role: str
    ssl_enabled: bool
    available_schemas: tuple[str, ...]


def _settings_url(settings: DatabaseSettings) -> URL:
    """Construye una URL SQLAlchemy sin convertir credenciales en texto plano."""

    return URL.create(
        "postgresql+psycopg2",
        username=settings.username,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        database=settings.database,
    )


def get_engine(settings: DatabaseSettings | Mapping[str, str] | None = None) -> Engine:
    """Crea un engine con pool acotado y compatibilidad temporal 4.x para mappings."""

    if settings is None:
        settings = load_database_settings(DatabaseProfile.TRAINING)
    if not isinstance(settings, DatabaseSettings):
        warnings.warn(
            "Dictionary DB configs are deprecated; use DatabaseSettings.",
            DeprecationWarning,
            stacklevel=2,
        )
        host = settings.get("host", "")
        sslmode = settings.get("sslmode", "disable" if host in _LOCAL_HOSTS else "require")
        settings = DatabaseSettings(
            profile=DatabaseProfile.TRAINING,
            host=host,
            port=int(settings.get("port", DEFAULT_DB_PORT)),
            database=settings.get("dbname", settings.get("database", "")),
            username=settings.get("user", settings.get("username", "")),
            password=settings.get("password", ""),
            sslmode=sslmode,
            sslrootcert=settings.get("sslrootcert"),
        )
    connect_args: dict[str, object] = {
        "connect_timeout": settings.connect_timeout_seconds,
        "application_name": settings.application,
        "sslmode": settings.sslmode,
    }
    if settings.sslrootcert:
        connect_args["sslrootcert"] = settings.sslrootcert
    return create_engine(
        _settings_url(settings),
        connect_args=connect_args,
        poolclass=QueuePool,
        pool_size=2,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=300,
        hide_parameters=True,
    )


@contextmanager
def database_engine(settings: DatabaseSettings) -> Iterator[Engine]:
    """Expone un engine comprobado y elimina certificados PEM temporales al cerrar."""

    engine = get_engine(settings)
    try:
        execute_with_retry(lambda: _probe_connection(engine))
        yield engine
    finally:
        engine.dispose()
        if settings._temporary_root_cert and settings.sslrootcert:
            Path(settings.sslrootcert).unlink(missing_ok=True)


def _probe_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def inspect_database(engine: Engine, profile: DatabaseProfile | str) -> DatabaseHealth:
    """Consulta información operativa sin leer tablas ni mostrar credenciales."""

    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT current_setting('server_version'), current_user, "
                "COALESCE((SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()), FALSE)"
            )
        ).one()
        available = tuple(
            schema
            for schema in DATABASE_SCHEMAS
            if connection.execute(text("SELECT to_regnamespace(:schema)"), {"schema": schema}).scalar()
        )
    url = engine.url
    return DatabaseHealth(
        profile=DatabaseProfile(profile).value,
        host=str(url.host or ""),
        port=int(url.port or DEFAULT_DB_PORT),
        database=str(url.database or ""),
        server_version=str(row[0]),
        current_role=str(row[1]),
        ssl_enabled=bool(row[2]),
        available_schemas=available,
    )


def test_connection(engine: Engine) -> bool:
    """Devuelve un diagnóstico booleano sin filtrar la excepción de infraestructura."""

    try:
        execute_with_retry(lambda: _probe_connection(engine))
        return True
    except OperationalError:  # pragma: no cover - servicio externo
        logger.warning("PostgreSQL connection test failed: OperationalError")
        return False
    except DatabaseOperationError:  # pragma: no cover - servicio externo
        logger.warning("PostgreSQL connection test failed after bounded retries")
        return False


def execute_with_retry(operation: Callable[[], object], *, attempts: int = 3) -> object:
    """Reintenta sólo fallos operativos transitorios y redacta su causa externa."""

    if attempts < 1:
        raise ValueError("Database retry attempts must be positive.")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OperationalError as exc:
            if attempt == attempts:
                raise DatabaseOperationError(
                    "PostgreSQL operation failed after bounded retries."
                ) from exc
            time.sleep(0.5 * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


__all__ = [
    "DatabaseHealth",
    "database_engine",
    "execute_with_retry",
    "get_engine",
    "inspect_database",
    "test_connection",
]
