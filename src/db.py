"""Database connection utilities for the VAAET intelligence layer.

Provides a reusable SQLAlchemy engine factory and helper functions used by
both the data-preparation and production notebooks.  Credentials are obtained
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
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB_ENV_VARS, DEFAULT_DB_PORT

__all__ = [
    "get_db_config",
    "get_engine",
    "load_from_backup",
    "load_telemetry",
    "parse_sql_dump",
    "restore_backup_to_sql",
    "test_connection",
]


# Credential helpers


def get_db_config() -> dict[str, str]:
    """Obtain database configuration from env vars or interactive input.

    Returns:
        Dictionary with keys: host, port, dbname, user, password.
    """
    config: dict[str, str] = {
        "host": os.environ.get("DB_HOST", ""),
        "port": os.environ.get("DB_PORT", DEFAULT_DB_PORT),
        "dbname": os.environ.get("DB_NAME", ""),
        "user": os.environ.get("DB_USER", ""),
        "password": os.environ.get("DB_PASSWORD", ""),
    }

    if not config["host"]:
        print("📋 PostgreSQL configuration (env vars not found)")
        config["host"] = input("   Host: ").strip()
        config["port"] = (
            input(f"   Port [{DEFAULT_DB_PORT}]: ").strip() or DEFAULT_DB_PORT
        )
        config["dbname"] = input("   Database: ").strip()
        config["user"] = input("   User: ").strip()
        config["password"] = getpass.getpass("   Password: ")

    return config


def _build_connection_string(config: dict[str, str]) -> str:
    """Build a PostgreSQL connection string from a config dict."""
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['dbname']}"
    )


# Engine factory


def get_engine(config: dict[str, str] | None = None) -> Engine:
    """Create a disposable SQLAlchemy engine.

    Args:
        config: Database config dict. If *None*, calls :func:`get_db_config`.

    Returns:
        A new :class:`sqlalchemy.engine.Engine`.
    """
    if config is None:
        config = get_db_config()
    return create_engine(_build_connection_string(config))


def test_connection(engine: Engine) -> bool:
    """Return *True* if the engine can connect to the database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"🔴 Connection test failed: {exc}")
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
    """Load raw telemetry from the ``traffic_data`` table.

    Args:
        config: DB credentials (used if *engine* is None).
        engine: Pre-existing SQLAlchemy engine.

    Returns:
        DataFrame with raw telemetry ordered by ``record_time``.
    """
    if engine is None:
        engine = get_engine(config)

    df: pd.DataFrame = pd.read_sql(text(TELEMETRY_QUERY), engine)
    engine.dispose()
    return df


# Backup restoration helpers


def restore_backup_to_sql(
    backup_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a binary ``pg_dump`` backup to a plain SQL text file.

    Requires ``pg_restore`` to be available on ``$PATH``.

    Args:
        backup_path: Path to the ``.backup`` (pg_dump custom format) file.
        output_path: Where to write the SQL text.  Defaults to
            ``<backup_path>.sql`` next to the original file.

    Returns:
        The *Path* to the generated SQL file.

    Raises:
        FileNotFoundError: If the backup file or ``pg_restore`` is missing.
        RuntimeError: If ``pg_restore`` exits with an error.
    """
    backup_path = Path(backup_path)
    if not backup_path.is_file():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    if shutil.which("pg_restore") is None:
        raise FileNotFoundError(
            "pg_restore not found. Install postgresql-client:\n"
            "  Colab/Ubuntu: !apt-get install -y postgresql-client\n"
            "  macOS:        brew install libpq\n"
            "  Windows:      install PostgreSQL or add bin/ to PATH"
        )

    if output_path is None:
        output_path = backup_path.with_suffix(".sql")
    output_path = Path(output_path)

    result = subprocess.run(
        [
            "pg_restore",
            "--no-owner",
            "--no-acl",
            "--no-comments",
            "-f",
            str(output_path),
            str(backup_path),
        ],
        capture_output=True,
        text=True,
    )
    # pg_restore returns 1 for warnings (e.g. missing role), which is fine
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"pg_restore failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"pg_restore produced an empty or missing file: {output_path}\n"
            f"stderr: {result.stderr}"
        )

    print(
        f"✅ Backup converted to SQL: {output_path} "
        f"({output_path.stat().st_size / 1024:.0f} KB)"
    )
    return output_path


# Column spec for the COPY block (matches TELEMETRY_QUERY)
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
    """Parse a plain-text SQL dump and extract ``traffic_data`` rows.

    Looks for ``COPY public.traffic_data (...) FROM stdin;`` blocks and
    reads the tab-separated rows until the ``\\.`` terminator.

    Args:
        sql_path: Path to the ``.sql`` file produced by :func:`restore_backup_to_sql`.

    Returns:
        DataFrame with the standard telemetry columns, typed appropriately.

    Raises:
        FileNotFoundError: If the SQL file does not exist.
        ValueError: If no ``traffic_data`` COPY block is found.
    """
    sql_path = Path(sql_path)
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    text_content = sql_path.read_text(encoding="utf-8", errors="replace")

    # Find COPY ... FROM stdin blocks for traffic_data
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
            # Extract column names from the COPY statement
            raw_cols = [c.strip() for c in match.group(1).split(",")]
            columns = raw_cols
            i += 1
            # Read data rows until the terminator
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

    # Parse tab-separated rows into a DataFrame
    assert columns is not None
    tsv = "\n".join(rows)
    df = pd.read_csv(
        io.StringIO(tsv),
        sep="\t",
        header=None,
        names=columns,
        na_values=["\\N"],
    )

    # Type casting (safe — columns may or may not be present)
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

    # Keep only the standard telemetry columns (in canonical order)
    available = [c for c in _TRAFFIC_DATA_COLUMNS if c in df.columns]
    df = df[available]

    print(f"✅ Parsed {len(df)} records from SQL dump ({len(columns)} columns)")
    return df


def load_from_backup(
    backup_path: str | Path,
    cache_csv: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end: restore a binary backup → parse → return DataFrame.

    Optionally saves a CSV cache so future runs skip ``pg_restore``.

    Args:
        backup_path: Path to the ``.backup`` file.
        cache_csv: If provided, the parsed DataFrame is saved here.

    Returns:
        DataFrame with standard telemetry columns.
    """
    backup_path = Path(backup_path)
    sql_path = restore_backup_to_sql(backup_path)

    try:
        df = parse_sql_dump(sql_path)
    finally:
        # Clean up the intermediate SQL file
        if sql_path.is_file():
            sql_path.unlink()
            print(f"🗑️  Temporary SQL file removed: {sql_path}")

    if cache_csv is not None:
        cache_csv = Path(cache_csv)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        print(f"💾 CSV cache saved: {cache_csv}")

    return df
