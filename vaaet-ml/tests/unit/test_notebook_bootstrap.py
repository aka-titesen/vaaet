from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ML_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = ML_ROOT / "scripts" / "notebook_bootstrap.py"


def _bootstrap_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vaaet_notebook_bootstrap", BOOTSTRAP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def install_spec(tmp_path: Path) -> object:
    module = _bootstrap_module()
    workspace = tmp_path / "vaaet"
    core = workspace / "vaaet-core"
    ml = workspace / "vaaet-ml"
    (workspace / ".git").mkdir(parents=True)
    core.mkdir()
    ml.mkdir()
    (core / "pyproject.toml").write_text("[project]\nname='core'\n", encoding="utf-8")
    (ml / "pyproject.toml").write_text("[project]\nname='ml'\n", encoding="utf-8")
    return module.NotebookInstallSpec(
        workspace_root=workspace,
        core_root=core,
        ml_root=ml,
        core_extras=("vision",),
        ml_extras=("database",),
        in_colab=True,
    )


def _successful_runner(calls: list[list[str]]):
    def runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    return runner


def test_first_install_forces_dependency_resolution(install_spec: object, tmp_path: Path) -> None:
    module = _bootstrap_module()
    calls: list[list[str]] = []

    module.install_notebook_components(
        install_spec, state_path=tmp_path / "state.json", runner=_successful_runner(calls)
    )

    assert len(calls) == 1
    assert "--force-reinstall" in calls[0]
    assert "--no-deps" not in calls[0]
    assert "--editable" not in calls[0]


def test_unchanged_dependencies_only_refresh_code(install_spec: object, tmp_path: Path) -> None:
    module = _bootstrap_module()
    state_path = tmp_path / "state.json"
    initial_calls: list[list[str]] = []
    module.install_notebook_components(
        install_spec, state_path=state_path, runner=_successful_runner(initial_calls)
    )
    refresh_calls: list[list[str]] = []

    module.install_notebook_components(
        install_spec, state_path=state_path, runner=_successful_runner(refresh_calls)
    )

    assert len(refresh_calls) == 1
    assert "--force-reinstall" in refresh_calls[0]
    assert "--no-deps" in refresh_calls[0]


def test_local_development_uses_editable_components(tmp_path: Path) -> None:
    module = _bootstrap_module()
    workspace = tmp_path / "vaaet"
    core = workspace / "vaaet-core"
    ml = workspace / "vaaet-ml"
    (workspace / ".git").mkdir(parents=True)
    core.mkdir()
    ml.mkdir()
    (core / "pyproject.toml").write_text("[project]\nname='core'\n", encoding="utf-8")
    (ml / "pyproject.toml").write_text("[project]\nname='ml'\n", encoding="utf-8")
    spec = module.NotebookInstallSpec(
        workspace_root=workspace,
        core_root=core,
        ml_root=ml,
        core_extras=("inference",),
        ml_extras=("training",),
        in_colab=False,
    )
    calls: list[list[str]] = []

    module.install_notebook_components(
        spec, state_path=tmp_path / "local-state.json", runner=_successful_runner(calls)
    )

    assert all(call.count("--editable") == 2 for call in calls)


def test_pyproject_change_resolves_dependencies_again(install_spec: object, tmp_path: Path) -> None:
    module = _bootstrap_module()
    state_path = tmp_path / "state.json"
    module.install_notebook_components(
        install_spec, state_path=state_path, runner=_successful_runner([])
    )
    install_spec.core_root.joinpath("pyproject.toml").write_text(
        "[project]\nname='core'\nversion='2'\n", encoding="utf-8"
    )
    calls: list[list[str]] = []

    module.install_notebook_components(
        install_spec, state_path=state_path, runner=_successful_runner(calls)
    )

    assert len(calls) == 1
    assert "--force-reinstall" in calls[0]
    assert "--no-deps" not in calls[0]


def test_install_failure_does_not_write_state(install_spec: object, tmp_path: Path) -> None:
    module = _bootstrap_module()
    state_path = tmp_path / "state.json"

    with pytest.raises(module.NotebookBootstrapError, match="instalación de VAAET falló"):
        module.install_notebook_components(
            install_spec,
            state_path=state_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
        )

    assert not state_path.exists()


def test_invalid_component_layout_fails_fast(tmp_path: Path) -> None:
    module = _bootstrap_module()
    workspace = tmp_path / "vaaet"
    (workspace / ".git").mkdir(parents=True)

    with pytest.raises(module.NotebookBootstrapError, match="pyproject.toml"):
        module.NotebookInstallSpec(
            workspace_root=workspace,
            core_root=workspace / "vaaet-core",
            ml_root=workspace / "vaaet-ml",
            core_extras=("vision",),
            ml_extras=("database",),
            in_colab=False,
        )
