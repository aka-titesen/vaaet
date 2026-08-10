"""Secure PostgreSQL configuration, connectivity, and portable backup readers."""

from __future__ import annotations

import io
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.pool import QueuePool

from vaaet.data.timestamps import normalize_timestamp_series
from vaaet.exceptions import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DatabaseNotConfiguredError,
)
from vaaet.logging import get_logger
from vaaet.settings import DATABASE_SCHEMAS, DEFAULT_DB_PORT

logger = get_logger(__name__)

RAW_TABLE = "vaaet_raw.traffic_data"
FEATURE_TABLE = "vaaet_ml.telemetry_features"
PREDICTION_TABLE = "vaaet_ml.traffic_predictions"
VALIDATION_TABLE = "vaaet_feedback.human_validations"
EFFECTIVE_LABELS_VIEW = "vaaet_feedback.effective_human_labels"

_RECOGNIZED_BACKUP_TABLES = {
    RAW_TABLE,
    FEATURE_TABLE,
    PREDICTION_TABLE,
    VALIDATION_TABLE,
    "public.traffic_data",
    "public.telemetry_raw",
    "public.traffic_classifications",
}
_TOC_TABLE_DATA_PATTERN = re.compile(
    r"^\s*(?P<dump_id>\d+);\s+\d+\s+\d+\s+TABLE DATA\s+"
    r"(?P<schema>\S+)\s+(?P<table>\S+)\s+.+$",
    re.IGNORECASE,
)
_COPY_PATTERN = re.compile(
    r"COPY\s+(?:(?P<schema>[\w\"]+)\.)?(?P<table>[\w\"]+)\s*"
    r"\((?P<columns>[^)]+)\)\s+FROM\s+stdin;",
    re.IGNORECASE,
)


class DatabaseProfile(str, Enum):
    """Least-privilege database identity used by a workflow."""

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
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


@dataclass(frozen=True, repr=False)
class DatabaseSettings:
    """Validated connection settings whose representation never exposes secrets."""

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
        return self.application_name or f"vaaet-{self.profile.value}-4.3.0"


@dataclass(frozen=True)
class DatabaseHealth:
    """Non-secret connection diagnostics safe to display in notebook output."""

    profile: str
    host: str
    port: int
    database: str
    server_version: str
    current_role: str
    ssl_enabled: bool
    available_schemas: tuple[str, ...]


@dataclass(frozen=True)
class _BackupCatalogEntry:
    """Exact TABLE DATA entry retained from a pg_restore table of contents."""

    qualified_name: str
    toc_line: str


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def _setting(name: str) -> str | None:
    """Read Colab Secrets first and process environment second."""
    return _colab_secret(name) or (os.environ.get(name) or "").strip() or None


def _materialize_root_certificate(pem: str) -> str:
    fd, path = tempfile.mkstemp(prefix="vaaet-postgres-ca-", suffix=".pem")
    try:
        os.write(fd, pem.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def load_database_settings(
    profile: DatabaseProfile | str,
    *,
    env_file: str | Path | None = None,
    allow_legacy: bool = True,
) -> DatabaseSettings:
    """Load a workflow profile from Colab Secrets or local environment.

    ``.env`` loading is explicit and unavailable as an implicit notebook side
    effect. Legacy ``DB_*`` names remain a deprecated VAAET 4.x fallback.
    """
    active_profile = DatabaseProfile(profile)
    if env_file is not None:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - optional dependency guard
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
    try:
        return load_database_settings(profile, env_file=env_file)
    except DatabaseNotConfiguredError:
        logger.info("Optional PostgreSQL profile=%s is not configured", DatabaseProfile(profile).value)
        return None


def load_reviewer_id() -> str:
    """Load the stable pseudonymous reviewer identifier without logging it."""
    reviewer_id = _setting("VAAET_REVIEWER_ID")
    if not reviewer_id:
        raise DatabaseNotConfiguredError(
            "VAAET_REVIEWER_ID is required in Colab Secrets or the local environment."
        )
    return reviewer_id


def _settings_url(settings: DatabaseSettings) -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=settings.username,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        database=settings.database,
    )


def get_engine(settings: DatabaseSettings | Mapping[str, str] | None = None) -> Engine:
    """Create a redacted, resilient SQLAlchemy engine.

    Mapping support is retained only for VAAET 4.x callers.
    """
    if settings is None:
        settings = load_database_settings(DatabaseProfile.TRAINING)
    if not isinstance(settings, DatabaseSettings):
        warnings.warn("Dictionary DB configs are deprecated; use DatabaseSettings.", DeprecationWarning)
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
    engine = get_engine(settings)
    try:
        def probe() -> None:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

        execute_with_retry(probe)
        yield engine
    finally:
        engine.dispose()
        if settings._temporary_root_cert and settings.sslrootcert:
            Path(settings.sslrootcert).unlink(missing_ok=True)


def inspect_database(engine: Engine, profile: DatabaseProfile | str) -> DatabaseHealth:
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
    try:
        def probe() -> None:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

        execute_with_retry(probe)
        return True
    except Exception as exc:  # pragma: no cover - depends on external service
        logger.warning("PostgreSQL connection test failed: %s", type(exc).__name__)
        return False


def execute_with_retry(operation, *, attempts: int = 3):
    """Retry a short idempotent database operation on transient connection loss."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OperationalError:
            if attempt == attempts:
                raise
            time.sleep(0.5 * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


TELEMETRY_QUERY = f"""
SELECT id, pipeline_run_id, clip_id, record_time, avg_speed,
       count_car, count_truck, count_bus, count_motorcycle, count_bicycle,
       total_vehicles, near_zero_motion_count, stationary_confirmed_count,
       rejected_speed_count, recovered_track_count, speed_sample_count,
       speed_measurement_quality, optical_flow_tracking_ratio,
       telemetry_schema_version
FROM {RAW_TABLE}
ORDER BY clip_id, record_time
"""

LEGACY_TELEMETRY_QUERY = """
SELECT id, clip_id, record_time, avg_speed, count_car, count_truck, count_bus,
       count_motorcycle, count_bicycle, total_vehicles
FROM public.traffic_data
ORDER BY clip_id, record_time
"""

HUMAN_GROUND_TRUTH_QUERY = f"""
SELECT id, source_record_id, pipeline_run_id, clip_id, record_time,
       feature_schema_version, avg_speed, total_vehicles, count_car,
       count_truck, count_bus, count_motorcycle, count_bicycle,
       heavy_vehicle_ratio, delta_speed, delta_count, transition_flag,
       speed_variance, cumulative_delta_speed, low_speed_persistence,
       speed_measurement_quality, optical_flow_tracking_ratio,
       near_zero_motion_ratio, stationary_confirmed_ratio,
       near_zero_motion_count, stationary_confirmed_count,
       rejected_speed_count, recovered_track_count, speed_sample_count,
       telemetry_schema_version, data_origin, synthetic_scenario,
       hour_of_day, weather_condition, created_at, prediction_id,
       model_version, traffic_state, is_human_validated, reviewer_id,
       reviewed_at, notes
FROM {EFFECTIVE_LABELS_VIEW}
ORDER BY clip_id, record_time
"""


def load_telemetry(
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load raw telemetry from the versioned schema, with a legacy view fallback."""
    owns_engine = engine is None
    active_engine = engine or get_engine(settings)
    try:
        try:
            return pd.read_sql(text(TELEMETRY_QUERY), active_engine)
        except ProgrammingError:
            legacy = pd.read_sql(text(LEGACY_TELEMETRY_QUERY), active_engine)
            for column in (
                "pipeline_run_id",
                "near_zero_motion_count",
                "stationary_confirmed_count",
                "rejected_speed_count",
                "recovered_track_count",
                "speed_sample_count",
                "speed_measurement_quality",
                "optical_flow_tracking_ratio",
                "telemetry_schema_version",
            ):
                legacy[column] = pd.NA
            return legacy
    finally:
        if owns_engine:
            active_engine.dispose()


def load_human_ground_truth(
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load only effective append-only human validations and their 19 features."""
    owns_engine = engine is None
    active_engine = engine or get_engine(settings)
    try:
        return pd.read_sql(text(HUMAN_GROUND_TRUTH_QUERY), active_engine)
    finally:
        if owns_engine:
            active_engine.dispose()


# Backup helpers
def _resolve_pg_restore(pg_restore_path: str | Path | None) -> Path:
    if pg_restore_path is None:
        discovered = shutil.which("pg_restore")
        if not discovered:
            raise ArtifactNotFoundError("pg_restore not found; install PostgreSQL client 17.")
        return Path(discovered).resolve()
    resolved = Path(pg_restore_path).expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactNotFoundError(f"Explicit pg_restore binary not found: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ArtifactValidationError(f"Explicit pg_restore path is not executable: {resolved}")
    return resolved


def _pg_restore_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        diagnostic = result.stderr.strip() or "no version output"
        raise ArtifactValidationError(
            f"Cannot execute PostgreSQL backup reader {binary}: {diagnostic}"
        )
    return version


def get_pg_restore_version(pg_restore_path: str | Path | None = None) -> str:
    """Return a validated, display-safe pg_restore version string."""
    return _pg_restore_version(_resolve_pg_restore(pg_restore_path))


def _read_backup_catalog(backup: Path, binary: Path) -> tuple[_BackupCatalogEntry, ...]:
    result = subprocess.run(
        [str(binary), "-l", str(backup)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ArtifactValidationError(
            f"Cannot inspect PostgreSQL backup with {binary}: {result.stderr.strip()}"
        )
    entries: list[_BackupCatalogEntry] = []
    for line in result.stdout.splitlines():
        match = _TOC_TABLE_DATA_PATTERN.match(line)
        if not match:
            continue
        schema = match.group("schema").strip(chr(34))
        table = match.group("table").strip(chr(34))
        entries.append(_BackupCatalogEntry(f"{schema}.{table}", line.strip()))
    return tuple(entries)


def inspect_backup_catalog(
    backup_path: str | Path,
    *,
    pg_restore_path: str | Path | None = None,
) -> tuple[str, ...]:
    backup = Path(backup_path)
    if not backup.is_file():
        raise ArtifactNotFoundError(f"Backup file not found: {backup}")
    binary = _resolve_pg_restore(pg_restore_path)
    entries = _read_backup_catalog(backup, binary)
    recognized = {
        entry.qualified_name
        for entry in entries
        if entry.qualified_name in _RECOGNIZED_BACKUP_TABLES
    }
    return tuple(sorted(recognized))


def restore_backup_to_sql(
    backup_path: str | Path,
    output_path: str | Path | None = None,
    pg_restore_path: str | Path | None = None,
    tables: Sequence[str] | None = None,
) -> Path:
    backup = Path(backup_path)
    if not backup.is_file():
        raise ArtifactNotFoundError(f"Backup file not found: {backup}")
    binary = _resolve_pg_restore(pg_restore_path)
    version = _pg_restore_version(binary)
    destination = Path(output_path) if output_path else backup.with_suffix(".sql")
    destination.unlink(missing_ok=True)
    command = [
        str(binary),
        "--data-only",
        "--no-owner",
        "--no-acl",
        "--no-comments",
    ]
    requested_tables = tuple(dict.fromkeys(tables or ()))
    toc_path: Path | None = None
    if requested_tables:
        entries = _read_backup_catalog(backup, binary)
        by_name = {entry.qualified_name: entry for entry in entries}
        missing = [table_name for table_name in requested_tables if table_name not in by_name]
        if missing:
            available = sorted(
                entry.qualified_name
                for entry in entries
                if entry.qualified_name in _RECOGNIZED_BACKUP_TABLES
            )
            raise ArtifactValidationError(
                "PostgreSQL backup selection did not match requested TABLE DATA entries: "
                f"missing={missing}, recognized={available}"
            )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".toc", delete=False
        ) as handle:
            handle.write("\n".join(by_name[name].toc_line for name in requested_tables))
            handle.write("\n")
            toc_path = Path(handle.name)
        command.extend(["--use-list", str(toc_path)])
    command.extend(["-f", str(destination), str(backup)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if toc_path is not None:
            toc_path.unlink(missing_ok=True)
    stderr = result.stderr.strip()
    if any(value in stderr.lower() for value in ("unsupported version", "unsupported archive")):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"pg_restore version mismatch ({version}, binary={binary}).\nstderr: {stderr}"
        )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"pg_restore failed (exit {result.returncode}, reader={version}, "
            f"tables={list(requested_tables)}):\n{stderr or 'no diagnostic output'}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"pg_restore produced an empty or missing file: {destination}")
    if requested_tables:
        extracted_tables: set[str] = set()
        with destination.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _COPY_PATTERN.match(line.strip())
                if match:
                    schema = (match.group("schema") or "public").strip(chr(34))
                    table = match.group("table").strip(chr(34))
                    extracted_tables.add(f"{schema}.{table}")
        missing_copy = [name for name in requested_tables if name not in extracted_tables]
        if missing_copy:
            destination.unlink(missing_ok=True)
            raise ArtifactValidationError(
                "PostgreSQL backup extraction produced no COPY block for requested tables: "
                f"{missing_copy} (reader={version})"
            )
    return destination


def parse_sql_dump_tables(sql_path: str | Path) -> dict[str, pd.DataFrame]:
    """Read recognized COPY blocks without executing dump SQL."""
    path = Path(sql_path)
    if not path.is_file():
        raise ArtifactNotFoundError(f"SQL file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: dict[str, pd.DataFrame] = {}
    index = 0
    while index < len(lines):
        match = _COPY_PATTERN.match(lines[index].strip())
        if not match:
            index += 1
            continue
        schema = (match.group("schema") or "public").strip('"')
        table_name = match.group("table").strip('"')
        columns = [column.strip().strip('"') for column in match.group("columns").split(",")]
        rows: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip() != "\\.":
            if lines[index]:
                rows.append(lines[index])
            index += 1
        qualified_name = f"{schema}.{table_name}"
        if rows:
            frames[qualified_name] = pd.read_csv(
                io.StringIO("\n".join(rows)),
                sep="\t",
                header=None,
                names=columns,
                na_values=["\\N"],
            )
        else:
            frames[qualified_name] = pd.DataFrame(columns=columns)
        index += 1
    return frames


def parse_sql_dump(sql_path: str | Path) -> pd.DataFrame:
    frames = parse_sql_dump_tables(sql_path)
    raw = frames.get(RAW_TABLE)
    if raw is None:
        raw = frames.get("public.traffic_data")
    if raw is None:
        raise ValueError("No COPY block for a recognized traffic_data table was found.")
    for column in raw.columns:
        if column == "record_time":
            raw[column] = normalize_timestamp_series(raw[column])
        elif column not in {"clip_id", "telemetry_schema_version", "pipeline_run_id"}:
            converted = pd.to_numeric(raw[column], errors="coerce")
            if raw[column].isna().equals(converted.isna()):
                raw[column] = converted
    return raw


def load_from_backup(
    backup_path: str | Path,
    cache_csv: str | Path | None = None,
    pg_restore_path: str | Path | None = None,
) -> pd.DataFrame:
    sql_path = restore_backup_to_sql(backup_path, pg_restore_path=pg_restore_path)
    try:
        frame = parse_sql_dump(sql_path)
    finally:
        sql_path.unlink(missing_ok=True)
    if cache_csv is not None:
        destination = Path(cache_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    return frame


__all__ = [
    "DatabaseHealth",
    "DatabaseProfile",
    "DatabaseSettings",
    "database_engine",
    "execute_with_retry",
    "get_pg_restore_version",
    "get_engine",
    "get_optional_database_settings",
    "inspect_backup_catalog",
    "inspect_database",
    "load_database_settings",
    "load_from_backup",
    "load_human_ground_truth",
    "load_reviewer_id",
    "load_telemetry",
    "parse_sql_dump",
    "parse_sql_dump_tables",
    "restore_backup_to_sql",
    "test_connection",
]
