from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vaaet import runtime
from vaaet.exceptions import RuntimeConfigurationError
from vaaet.runtime import (
    _framework_gpu_available,
    _nvidia_smi,
    _pip_check,
    _validate_package_origin,
    _validate_python_version,
)


def test_validate_python_version_accepts_supported_bounds() -> None:
    _validate_python_version((3, 10))
    _validate_python_version((3, 13))


def test_validate_python_version_rejects_unsupported_runtime() -> None:
    with pytest.raises(RuntimeConfigurationError, match="supports Python"):
        _validate_python_version((3, 14))


def test_package_origin_rejects_namespace_package(tmp_path: Path) -> None:
    package = SimpleNamespace(__file__=None, __path__=[str(tmp_path / "vaaet")])
    with pytest.raises(RuntimeConfigurationError, match="namespace package"):
        _validate_package_origin(package, tmp_path, True)


def test_package_origin_accepts_colab_wheel(tmp_path: Path) -> None:
    package_file = tmp_path / "site-packages" / "vaaet" / "__init__.py"
    package = SimpleNamespace(__file__=str(package_file), __path__=[str(package_file.parent)])
    assert _validate_package_origin(package, tmp_path / "workspace" / "vaaet-ml", True) == package_file.resolve()


def test_gpu_preflight_rejects_an_unknown_framework() -> None:
    with pytest.raises(RuntimeConfigurationError, match="Unsupported runtime framework"):
        _framework_gpu_available("jax")


def test_nvidia_smi_uses_a_bounded_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0, stdout="T4, 15360, 42\n"),
    )

    assert _nvidia_smi() == "T4, 15360, 42"
    assert calls == [["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]]


def test_pip_check_keeps_diagnostics_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="resolver warning\n", stderr="conflict\n"),
    )

    assert _pip_check() == "resolver warning\nconflict"
