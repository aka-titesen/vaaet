"""Tests for vaaet.data.database utilities.

Only pure functions are tested. Actual DB connections require credentials
and are marked with ``@pytest.mark.db`` (skipped by default).
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pandas as pd
import pytest

from vaaet.data.database import _build_connection_string


def test_hydrate_db_environment_from_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    from vaaet.data.database import hydrate_db_environment_from_colab

    class Secrets:
        @staticmethod
        def get(name: str) -> str:
            return f"secret-{name.lower()}"

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = Secrets()
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    for name in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    loaded = hydrate_db_environment_from_colab()

    assert loaded == ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


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
        from vaaet.data.database import get_db_config

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
        from vaaet.data.database import get_db_config
        from vaaet.settings import DEFAULT_DB_PORT

        monkeypatch.setenv("DB_HOST", "host")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "pass")
        monkeypatch.delenv("DB_PORT", raising=False)

        config = get_db_config()
        assert config["port"] == DEFAULT_DB_PORT

    def test_non_interactive_raises_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """interactive=False must raise RuntimeError when env vars are missing."""
        from vaaet.data.database import get_db_config

        for var in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(RuntimeError, match="DB credentials not found"):
            get_db_config(interactive=False)

    def test_non_interactive_succeeds_with_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """interactive=False must succeed when env vars are set."""
        from vaaet.data.database import get_db_config

        monkeypatch.setenv("DB_HOST", "h")
        monkeypatch.setenv("DB_NAME", "d")
        monkeypatch.setenv("DB_USER", "u")
        monkeypatch.setenv("DB_PASSWORD", "p")

        config = get_db_config(interactive=False)
        assert config["host"] == "h"


# Backup restoration tests


class TestRestoreBackupToSql:
    """Test restore_backup_to_sql() error handling (no pg_restore needed)."""

    def test_missing_backup_file(self, tmp_path: Path) -> None:
        from vaaet.data.database import restore_backup_to_sql

        with pytest.raises(FileNotFoundError, match="Backup file not found"):
            restore_backup_to_sql(tmp_path / "nonexistent.backup")

    def test_missing_pg_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vaaet.data.database import restore_backup_to_sql

        # Create a dummy backup file
        backup = tmp_path / "test.backup"
        backup.write_bytes(b"\x00" * 100)

        # Hide pg_restore from PATH
        monkeypatch.setenv("PATH", str(tmp_path))

        with pytest.raises(FileNotFoundError, match="pg_restore not found"):
            restore_backup_to_sql(backup)

    def test_missing_explicit_pg_restore(self, tmp_path: Path) -> None:
        from vaaet.data.database import restore_backup_to_sql

        backup = tmp_path / "test.backup"
        backup.write_bytes(b"PGDMP")

        with pytest.raises(FileNotFoundError, match="Explicit pg_restore binary not found"):
            restore_backup_to_sql(
                backup,
                pg_restore_path=tmp_path / "missing-pg_restore",
            )

    def test_rejects_non_executable_explicit_pg_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vaaet.data.database import restore_backup_to_sql

        backup = tmp_path / "test.backup"
        backup.write_bytes(b"PGDMP")
        pg_restore = tmp_path / "pg_restore"
        pg_restore.write_text("binary", encoding="utf-8")
        monkeypatch.setattr("vaaet.data.database.os.access", lambda *_: False)

        with pytest.raises(ValueError, match="not executable"):
            restore_backup_to_sql(backup, pg_restore_path=pg_restore)

    def test_uses_explicit_pg_restore_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        from vaaet.data.database import restore_backup_to_sql

        backup = tmp_path / "test.backup"
        backup.write_bytes(b"PGDMP")
        pg_restore = tmp_path / "pg_restore-17"
        pg_restore.write_text("binary", encoding="utf-8")
        output = tmp_path / "dump.sql"
        calls: list[list[str]] = []

        monkeypatch.setattr("vaaet.data.database.os.access", lambda *_: True)

        def fake_run(command, **_kwargs):
            calls.append(command)
            if "--version" in command:
                return MagicMock(returncode=0, stdout="pg_restore (PostgreSQL) 17.6", stderr="")
            output.write_text("-- restored", encoding="utf-8")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("vaaet.data.database.subprocess.run", fake_run)

        result = restore_backup_to_sql(
            backup,
            output_path=output,
            pg_restore_path=pg_restore,
        )

        expected = str(pg_restore.resolve())
        assert result == output
        assert calls[0][0] == expected
        assert calls[1][0] == expected

    def test_version_mismatch_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pg_restore stderr with 'unsupported version' must raise RuntimeError."""
        from unittest.mock import MagicMock, patch

        from vaaet.data.database import restore_backup_to_sql

        backup = tmp_path / "test.backup"
        backup.write_bytes(b"\x00" * 100)

        # Mock shutil.which to return a fake pg_restore path
        monkeypatch.setattr("vaaet.data.database.shutil.which", lambda _: "/usr/bin/pg_restore")

        # Mock subprocess.run to simulate version mismatch
        ver_result = MagicMock(returncode=0, stdout="pg_restore (PostgreSQL) 14.0")
        run_result = MagicMock(
            returncode=1,
            stderr="pg_restore: error: unsupported version (1.16) in file header",
        )

        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # --version call
                return ver_result
            return run_result  # actual pg_restore call

        with patch("vaaet.data.database.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match=r"version mismatch .*binary="):
                restore_backup_to_sql(backup)


def test_load_from_backup_propagates_explicit_pg_restore_path(tmp_path: Path) -> None:
    from unittest.mock import patch

    from vaaet.data.database import load_from_backup

    backup = tmp_path / "traffic_data.backup"
    backup.write_bytes(b"PGDMP")
    sql_path = tmp_path / "traffic_data.sql"
    sql_path.write_text("-- restored", encoding="utf-8")
    pg_restore = tmp_path / "pg_restore-17"

    expected = pd.DataFrame({"id": [1]})
    with (
        patch("vaaet.data.database.restore_backup_to_sql", return_value=sql_path) as restore,
        patch("vaaet.data.database.parse_sql_dump", return_value=expected),
    ):
        result = load_from_backup(backup, pg_restore_path=pg_restore)

    restore.assert_called_once_with(backup, pg_restore_path=pg_restore)
    pd.testing.assert_frame_equal(result, expected)


class TestParseSqlDump:
    """Test parse_sql_dump() with synthetic SQL text."""

    @pytest.fixture()
    def sample_sql(self, tmp_path: Path) -> Path:
        """Create a minimal SQL dump with a COPY block."""
        content = textwrap.dedent("""\
            --
            -- PostgreSQL database dump
            --
            SET statement_timeout = 0;

            COPY public.traffic_data (id, clip_id, record_time, avg_speed, count_car, count_truck, count_bus, count_motorcycle, count_bicycle, total_vehicles) FROM stdin;
            1\tclip_001\t2024-01-15 08:00:00\t55.3\t10\t3\t1\t2\t0\t16
            2\tclip_001\t2024-01-15 08:01:00\t48.7\t8\t2\t0\t1\t1\t12
            3\tclip_002\t2024-01-15 08:02:00\t12.1\t15\t5\t2\t3\t0\t25
            \\.

            -- end
        """)
        sql_file = tmp_path / "dump.sql"
        sql_file.write_text(content, encoding="utf-8")
        return sql_file

    def test_parses_correct_row_count(self, sample_sql: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        df = parse_sql_dump(sample_sql)
        assert len(df) == 3

    def test_parses_correct_columns(self, sample_sql: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        df = parse_sql_dump(sample_sql)
        expected = {
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
        }
        assert set(df.columns) == expected

    def test_numeric_types(self, sample_sql: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        df = parse_sql_dump(sample_sql)
        assert pd.api.types.is_numeric_dtype(df["avg_speed"])
        assert pd.api.types.is_numeric_dtype(df["count_car"])
        assert pd.api.types.is_numeric_dtype(df["total_vehicles"])

    def test_datetime_parsing(self, sample_sql: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        df = parse_sql_dump(sample_sql)
        assert pd.api.types.is_datetime64_any_dtype(df["record_time"])

    def test_speed_values(self, sample_sql: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        df = parse_sql_dump(sample_sql)
        assert df["avg_speed"].iloc[0] == pytest.approx(55.3)
        assert df["avg_speed"].iloc[2] == pytest.approx(12.1)

    def test_missing_file(self, tmp_path: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        with pytest.raises(FileNotFoundError, match="SQL file not found"):
            parse_sql_dump(tmp_path / "nonexistent.sql")

    def test_no_copy_block(self, tmp_path: Path) -> None:
        from vaaet.data.database import parse_sql_dump

        sql_file = tmp_path / "empty.sql"
        sql_file.write_text("-- empty dump\nSELECT 1;\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No COPY block"):
            parse_sql_dump(sql_file)

    def test_null_handling(self, tmp_path: Path) -> None:
        """Postgres NULL (\\N) should be parsed as NaN/NA."""
        from vaaet.data.database import parse_sql_dump

        content = textwrap.dedent("""\
            COPY public.traffic_data (id, clip_id, record_time, avg_speed, count_car, count_truck, count_bus, count_motorcycle, count_bicycle, total_vehicles) FROM stdin;
            1\tclip_001\t2024-01-15 08:00:00\t\\N\t10\t3\t1\t2\t0\t16
            \\.
        """)
        sql_file = tmp_path / "nulls.sql"
        sql_file.write_text(content, encoding="utf-8")
        df = parse_sql_dump(sql_file)
        assert pd.isna(df["avg_speed"].iloc[0])
