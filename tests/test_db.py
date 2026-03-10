"""Tests for src/db.py — database utilities.

Only pure functions are tested. Actual DB connections require credentials
and are marked with ``@pytest.mark.db`` (skipped by default).
"""

from __future__ import annotations

import os

import pytest

from src.db import _build_connection_string


class TestBuildConnectionString:
    """Validate the connection string builder (pure function)."""

    def test_standard_config(self) -> None:
        config = {
            "user": "admin",
            "password": "secret",
            "host": "db.example.com",
            "port": "5432",
            "dbname": "traffic",
        }
        result = _build_connection_string(config)
        assert result == "postgresql://admin:secret@db.example.com:5432/traffic"

    def test_special_chars_in_password(self) -> None:
        config = {
            "user": "admin",
            "password": "p@ss:w0rd",
            "host": "localhost",
            "port": "5432",
            "dbname": "test",
        }
        result = _build_connection_string(config)
        assert "p@ss:w0rd" in result

    def test_custom_port(self) -> None:
        config = {
            "user": "u",
            "password": "p",
            "host": "h",
            "port": "15432",
            "dbname": "d",
        }
        result = _build_connection_string(config)
        assert ":15432/" in result


class TestGetDbConfig:
    """Test get_db_config() when environment variables are set."""

    def test_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.db import get_db_config

        monkeypatch.setenv("DB_HOST", "test-host")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "test-db")
        monkeypatch.setenv("DB_USER", "test-user")
        monkeypatch.setenv("DB_PASSWORD", "test-pass")

        config = get_db_config()
        assert config["host"] == "test-host"
        assert config["port"] == "5433"
        assert config["dbname"] == "test-db"
        assert config["user"] == "test-user"
        assert config["password"] == "test-pass"

    def test_default_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config import DEFAULT_DB_PORT
        from src.db import get_db_config

        monkeypatch.setenv("DB_HOST", "host")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.delenv("DB_PORT", raising=False)

        config = get_db_config()
        assert config["port"] == DEFAULT_DB_PORT
