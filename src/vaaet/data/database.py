"""Database connection utilities for the VAAET intelligence layer.

Provides a reusable SQLAlchemy engine factory and helper functions used by
both the data-preparation and production notebooks. Credentials are obtained
from environment variables or interactive input — never hard-coded.
"""

from __future__ import annotations

import getpass
import io
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB_ENV_VARS, DEFAULT_DB_PORT
from src.exceptions import ArtifactNotFoundError, DatabaseNotConfiguredError
from src.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_db_config",
    "get_engine",
    "get_optional_db_config",
    "load_from_backup",
    "load_telemetry",
    "parse_sql_dump",
    "restore_backup_to_sql",
    "test_connection",
]


# Credential helpers


def get_db_config(*, interactive: bool = True) -> dict[str, str]:
    """Obtain database configuration from env vars or interactive input."""
    config: dict[str, str] = {
        "host": os.environ.get("DB_HOST", ""),
        "port": os.environ.get("DB_PORT", DEFAULT_DB_PORT),
        "dbname": os.environ.get("DB_NAME", ""),
        "user": os.environ.get("DB_USER", ""),
        "password": os.environ.get("DB_PASSWORD", ""),
    }

    if not config["host"]:
        if not interactive:
            raise DatabaseNotConfiguredError(
                "DB credentials not found in environment variables "
                "(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD). "
                "Set them or use interactive=True to prompt for input."
            )
        logger.info("PostgreSQL configuration not found in env vars; prompting interactively")
        config["host"] = input("Host: ").strip()
        config["port"] = input(f"Port [{DEFAULT_DB_PORT}]: ").strip() or DEFAULT_DB_PORT
        config["dbname"] = input("Database: ").strip()
        config["user"] = input("User: ").strip()
        config["password"] = getpass.getpass("Password: ")

    return config


def get_optional_db_config(*, interactive: bool = False) -> dict[str, str] | None:
    """Return DB config when available, otherwise ``None`` without raising."""
    try:
        return get_db_config(interactive=interactive)
    except DatabaseNotConfiguredError:
        logger.info("Optional database configuration not found; continuing without DB")
        return None


def _build_connection_string(config: dict[str, str]) -> str:
    """Build a PostgreSQL connection string from a config dict."""
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['dbname']}"
    )


# Engine factory


def get_engine(config: dict[str, str] | None = None) -> Engine:
    """Create a disposable SQLAlchemy engine."""
    if config is None:
        config = get_db_config()
    return create_engine(_build_connection_string(config))


def test_connection(engine: Engine) -> bool:
    """Return ``True`` if the engine can connect to the database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on real DB
        logger.warning("Connection test failed: %s", exc)
        return False


# Data loaders


TELEMETRY_QUERY: str = """
    SELECT id, clip_id, record_time, avg_speed,
           count_car, count_truck, count_bus,
           count_motorcycle, count_bicycle, total_vehicles
    FROM traffic_data
    ORDER BY record_time
"""


def load_telemetry(
    config: dict[str, str] | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load raw telemetry from the ``traffic_data`` table."""
    owns_engine = engine is None
    active_engine = engine or get_engine(config)

    try:
        return pd.read_sql(text(TELEMETRY_QUERY), active_engine)
    finally:
        if owns_engine:
            active_engine.dispose()


# Backup restoration helpers


def restore_backup_to_sql(
    backup_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a binary ``pg_dump`` backup to a plain SQL text file."""
    backup_path = Path(backup_path)
    if not backup_path.is_file():
        raise ArtifactNotFoundError(f"Backup file not found: {backup_path}")

    pg_restore_path = shutil.which("pg_restore")
    if pg_restore_path is None:
        raise ArtifactNotFoundError(
            "pg_restore not found. Install postgresql-client:\n"
            "  Colab/Ubuntu: !apt-get install -y postgresql-client\n"
            "  macOS:        brew install libpq\n"
            "  Windows:      install PostgreSQL or add bin/ to PATH"
        )

    ver_result = subprocess.run(
        [pg_restore_path, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    pg_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "unknown"
    logger.info("Using %s", pg_version)

    if output_path is None:
        output_path = backup_path.with_suffix(".sql")
    output_path = Path(output_path)

    result = subprocess.run(
        [
            pg_restore_path,
            "--no-owner",
            "--no-acl",
            "--no-comments",
            "-f",
            str(output_path),
            str(backup_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    stderr = result.stderr.strip()
    version_error_patterns = [
        "unsupported version",
        "unsupported archive",
        "unrecognized archive format",
    ]
    if any(pattern in stderr.lower() for pattern in version_error_patterns):
        raise RuntimeError(
            f"pg_restore version mismatch ({pg_version}).\n"
            "The backup was created with a newer PostgreSQL version.\n"
            "Install a newer postgresql-client (e.g. postgresql-client-17).\n"
            f"stderr: {stderr}"
        )

    if result.returncode not in (0, 1):
        raise RuntimeError(f"pg_restore failed (exit {result.returncode}):\n{stderr}")

    if stderr and result.returncode == 1:
        logger.warning("pg_restore warnings: %s", stderr[:200])

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"pg_restore produced an empty or missing file: {output_path}\n"
            f"stderr: {stderr}"
        )

    logger.info(
        "Backup converted to SQL: %s (%.0f KB)",
        output_path,
        output_path.stat().st_size / 1024,
    )
    return output_path


_TRAFFIC_DATA_COLUMNS: list[str] = [
    "id",
    "clip_id",
    "record_time",
    "avg_speed",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "total_vehicles",
]


def parse_sql_dump(sql_path: str | Path) -> pd.DataFrame:
    """Parse a plain-text SQL dump and extract ``traffic_data`` rows."""
    sql_path = Path(sql_path)
    if not sql_path.is_file():
        raise ArtifactNotFoundError(f"SQL file not found: {sql_path}")

    text_content = sql_path.read_text(encoding="utf-8", errors="replace")
    copy_pattern = re.compile(
        r"COPY\s+(?:public\.)?traffic_data\s*\(([^)]+)\)\s+FROM\s+stdin;",
        re.IGNORECASE,
    )

    rows: list[str] = []
    columns: list[str] | None = None
    lines = text_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = copy_pattern.match(line.strip())
        if match:
            columns = [col.strip() for col in match.group(1).split(",")]
            i += 1
            while i < len(lines) and lines[i].strip() != "\\.":
                row = lines[i].strip()
                if row:
                    rows.append(row)
                i += 1
        i += 1

    if not rows:
        raise ValueError(
            f"No COPY block for traffic_data found in {sql_path}. "
            "Ensure the backup contains the traffic_data table."
        )

    assert columns is not None
    tsv = "\n".join(rows)
    df = pd.read_csv(
        io.StringIO(tsv),
        sep="\t",
        header=None,
        names=columns,
        na_values=["\\N"],
    )

    int_cols = [
        "id",
        "count_car",
        "count_truck",
        "count_bus",
        "count_motorcycle",
        "count_bicycle",
        "total_vehicles",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "avg_speed" in df.columns:
        df["avg_speed"] = pd.to_numeric(df["avg_speed"], errors="coerce")

    if "record_time" in df.columns:
        df["record_time"] = pd.to_datetime(df["record_time"], errors="coerce")

    available = [c for c in _TRAFFIC_DATA_COLUMNS if c in df.columns]
    df = df[available]
    logger.info("Parsed %s records from SQL dump", len(df))
    return df


def load_from_backup(
    backup_path: str | Path,
    cache_csv: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end: restore a binary backup → parse → return DataFrame."""
    backup_path = Path(backup_path)
    sql_path = restore_backup_to_sql(backup_path)

    try:
        df = parse_sql_dump(sql_path)
    finally:
        if sql_path.is_file():
            sql_path.unlink()
            logger.info("Temporary SQL file removed: %s", sql_path)

    if cache_csv is not None:
        cache_csv = Path(cache_csv)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        logger.info("CSV cache saved: %s", cache_csv)

    return df
