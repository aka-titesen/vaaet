"""Idempotent, notebook-safe runtime diagnostics for local development and Colab."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from vaaet.exceptions import RuntimeConfigurationError

_SUPPORTED_PYTHON_MIN = (3, 10)
_SUPPORTED_PYTHON_MAX = (3, 13)
_GIB = 1024**3


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Redacted and bounded evidence collected before an expensive workflow."""

    in_colab: bool
    workspace_root: Path
    core_root: Path
    ml_root: Path
    package_file: Path
    ml_package_file: Path
    git_commit: str
    python_version: str
    framework: str | None
    framework_gpu_available: bool | None
    nvidia_smi: str | None
    total_ram_gib: float | None
    available_ram_gib: float | None
    content_free_gib: float | None
    pip_check_output: str | None


def _validate_python_version(version: tuple[int, int]) -> None:
    if not _SUPPORTED_PYTHON_MIN <= version <= _SUPPORTED_PYTHON_MAX:
        raise RuntimeConfigurationError(
            "VAAET supports Python 3.10–3.13. In Colab, select a compatible runtime "
            "before installing workflow extras."
        )


def _clear_vaaet_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name in {"vaaet", "vaaet_ml"} or module_name.startswith(
            ("vaaet.", "vaaet_ml.")
        ):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _validate_package_origin(
    package: ModuleType,
    project_root: Path,
    import_name: str,
    in_colab: bool,
) -> Path:
    package_file = getattr(package, "__file__", None)
    if not package_file:
        resolved_paths = list(getattr(package, "__path__", ()))
        raise RuntimeConfigurationError(
            f"The '{import_name}' import resolved to a namespace package; rerun the environment cell. "
            f"Resolved locations: {resolved_paths}"
        )
    origin = Path(package_file).resolve()
    expected_editable_root = (project_root / "src" / import_name).resolve()
    if in_colab and project_root.resolve() in origin.parents:
        raise RuntimeConfigurationError(
            f"Colab must load installed {import_name}, not the checkout: {origin}"
        )
    if not in_colab and origin.parent != expected_editable_root:
        raise RuntimeConfigurationError(
            f"Local editable installation has an unexpected origin: {origin}"
        )
    return origin


def _git_commit(workspace_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace_root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeConfigurationError("Unable to determine the checked-out VAAET commit.")
    return result.stdout.strip()


def _pip_check() -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return None if result.returncode == 0 else output or "pip check returned no diagnostics"


def _nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _memory_diagnostics() -> tuple[float | None, float | None]:
    if not hasattr(os, "sysconf"):
        return None, None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError):
        return None, None
    return total_pages * page_size / _GIB, available_pages * page_size / _GIB


def _framework_gpu_available(framework: str | None) -> bool | None:
    if framework is None:
        return None
    if framework == "tensorflow":
        import tensorflow as tf

        return bool(tf.config.list_physical_devices("GPU"))
    if framework == "torch":
        import torch

        return bool(torch.cuda.is_available())
    raise RuntimeConfigurationError(f"Unsupported runtime framework: {framework!r}.")


def bootstrap_notebook_runtime(
    *,
    workspace_root: Path,
    core_root: Path,
    ml_root: Path,
    in_colab: bool,
    framework: str | None,
    require_gpu: bool,
) -> RuntimeDiagnostics:
    """Validate the installed package and preflight the runtime without exposing secrets."""

    _validate_python_version((sys.version_info.major, sys.version_info.minor))
    if not (workspace_root / ".git").is_dir() or not all(
        path.joinpath("pyproject.toml").is_file() for path in (core_root, ml_root)
    ):
        raise RuntimeConfigurationError("VAAET workspace, core, or ML component was not found.")

    _clear_vaaet_modules()
    import vaaet

    import vaaet_ml

    package_file = _validate_package_origin(vaaet, core_root, "vaaet", in_colab)
    ml_package_file = _validate_package_origin(vaaet_ml, ml_root, "vaaet_ml", in_colab)
    framework_gpu_available = _framework_gpu_available(framework)
    if require_gpu and not framework_gpu_available:
        raise RuntimeConfigurationError(
            "A GPU is required for this Colab workflow. Select a GPU runtime and rerun "
            "the environment cell; VAAET will not start a long CPU fallback."
        )

    total_ram_gib, available_ram_gib = _memory_diagnostics()
    content_root = Path("/content") if in_colab else ml_root
    content_free_gib = shutil.disk_usage(content_root).free / _GIB
    diagnostics = RuntimeDiagnostics(
        in_colab=in_colab,
        workspace_root=workspace_root.resolve(),
        core_root=core_root.resolve(),
        ml_root=ml_root.resolve(),
        package_file=package_file,
        ml_package_file=ml_package_file,
        git_commit=_git_commit(workspace_root),
        python_version=sys.version.split()[0],
        framework=framework,
        framework_gpu_available=framework_gpu_available,
        nvidia_smi=_nvidia_smi(),
        total_ram_gib=total_ram_gib,
        available_ram_gib=available_ram_gib,
        content_free_gib=content_free_gib,
        pip_check_output=_pip_check(),
    )
    print_runtime_diagnostics(diagnostics)
    return diagnostics


def print_runtime_diagnostics(diagnostics: RuntimeDiagnostics) -> None:
    """Display bounded, non-sensitive runtime evidence for notebook users."""

    print(
        "✅ Runtime ready | "
        f"Python {diagnostics.python_version} | commit={diagnostics.git_commit} | "
        f"package={diagnostics.package_file}"
    )
    if diagnostics.framework is not None:
        print(
            f"Framework GPU ({diagnostics.framework}): "
            f"{bool(diagnostics.framework_gpu_available)}"
        )
    if diagnostics.nvidia_smi:
        print(f"nvidia-smi: {diagnostics.nvidia_smi}")
    if diagnostics.total_ram_gib is not None and diagnostics.available_ram_gib is not None:
        print(
            f"RAM total/free: {diagnostics.total_ram_gib:.1f}/{diagnostics.available_ram_gib:.1f} GiB "
            "(keep active workflow memory below 11 GiB)"
        )
    print(f"Local staging free: {diagnostics.content_free_gib:.1f} GiB")
    if diagnostics.pip_check_output:
        print("⚠️ pip check reported managed-runtime conflicts; workflow imports remain explicit.")
        print(diagnostics.pip_check_output)
