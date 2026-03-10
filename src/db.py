"""Database connection utilities for the VAAET intelligence layer.

Provides a reusable SQLAlchemy engine factory and helper functions used by
both the data-preparation and production notebooks.  Credentials are obtained
from environment variables or interactive input — never hard-coded.
"""

from __future__ import annotations

import getpass
import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB_ENV_VARS, DEFAULT_DB_PORT

__all__ = [
    "get_db_config",
    "get_engine",
    "load_telemetry",
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
