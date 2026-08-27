"""Architectural boundary checks for the portable VAAET core."""

from __future__ import annotations

from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORTS = (
    "vaaet_ml",
    "sqlalchemy",
    "psycopg",
    "google.colab",
    "dvc",
    "pydrive",
)


def test_core_has_no_lab_storage_or_notebook_dependencies() -> None:
    source_files = CORE_ROOT.joinpath("src", "vaaet").rglob("*.py")
    violations: list[str] = []
    for path in source_files:
        content = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IMPORTS:
            if forbidden in content:
                violations.append(f"{path.relative_to(CORE_ROOT)} -> {forbidden}")
    assert violations == []


def test_core_distribution_declares_only_portable_base_dependencies() -> None:
    pyproject = (CORE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "vaaet-core"' in pyproject
    assert "sqlalchemy" not in pyproject
    assert "psycopg" not in pyproject
    assert "dvc" not in pyproject
