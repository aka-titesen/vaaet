"""Verify that all notebooks orchestrate shared package APIs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "collection": REPO_ROOT / "notebooks/data-collection/collect_traffic_telemetry.ipynb",
    "training": REPO_ROOT / "notebooks/training/train_traffic_state_classifier.ipynb",
    "inference": REPO_ROOT / "notebooks/inference/analyze_traffic_video.ipynb",
}


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("path", NOTEBOOKS.values())
def test_notebook_has_one_editable_install_and_no_import_hacks(path: Path) -> None:
    code = _code(path)
    assert code.count('"-e"') == 1
    assert "sys.path.insert" not in code
    assert "install_if_missing" not in code


@pytest.mark.parametrize("path", NOTEBOOKS.values())
def test_colab_uses_wheel_and_local_uses_editable_install(path: Path) -> None:
    code = _code(path)
    assert "install_command.append" in code
    assert "install_command.extend" in code
    assert code.count('"-e"') == 1
    assert code.index("if IN_COLAB:") < code.index("install_command.append")


@pytest.mark.parametrize("path", NOTEBOOKS.values())
def test_notebook_clears_cache_and_validates_package_origin(path: Path) -> None:
    code = _code(path)
    assert 'module_name == "vaaet"' in code
    assert 'module_name.startswith("vaaet.")' in code
    assert "importlib.invalidate_caches()" in code
    assert "def validate_vaaet_origin(" in code
    assert "VAAET_PACKAGE_FILE = validate_vaaet_origin" in code


def _load_origin_validator() -> object:
    tree = ast.parse(_code(NOTEBOOKS["collection"]))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_vaaet_origin"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {"Path": Path}
    exec(compile(module, "validate_vaaet_origin", "exec"), namespace)
    return namespace["validate_vaaet_origin"]


def test_origin_validator_rejects_namespace_package(tmp_path: Path) -> None:
    validate = _load_origin_validator()
    package = SimpleNamespace(__file__=None, __path__=[str(tmp_path / "vaaet")])
    with pytest.raises(ImportError, match="namespace package"):
        validate(package, tmp_path / "repository", True)  # type: ignore[operator]


def test_origin_validator_accepts_wheel_in_colab(tmp_path: Path) -> None:
    validate = _load_origin_validator()
    package_file = tmp_path / "site-packages/vaaet/__init__.py"
    package = SimpleNamespace(__file__=str(package_file), __path__=[str(package_file.parent)])
    result = validate(package, tmp_path / "repository", True)  # type: ignore[operator]
    assert result == package_file.resolve()


@pytest.mark.parametrize("path", NOTEBOOKS.values())
def test_notebook_pip_check_is_visible_but_non_blocking(path: Path) -> None:
    code = _code(path)
    assert code.count('"pip", "check"') == 1
    assert 'check_call([sys.executable, "-m", "pip", "check"])' not in code
    assert "capture_output=True" in code
    assert "check=False" in code
    assert "pip_check.returncode" in code
    assert "print(pip_check_output" in code


def test_notebooks_keep_workflow_smoke_imports() -> None:
    expected_imports = {
        "collection": ("cv2", "numpy", "pandas", "psycopg2", "sqlalchemy", "torch", "ultralytics"),
        "training": ("imblearn", "joblib", "numpy", "pandas", "psycopg2", "sqlalchemy", "tensorflow"),
        "inference": ("cv2", "joblib", "numpy", "pandas", "psycopg2", "sqlalchemy", "tensorflow", "ultralytics"),
    }
    for workflow, import_names in expected_imports.items():
        code = _code(NOTEBOOKS[workflow])
        for import_name in import_names:
            assert import_name in code


def test_collection_uses_shared_analysis_and_data_contracts() -> None:
    code = _code(NOTEBOOKS["collection"])
    assert "from vaaet.vision.analysis import analyze_video" in code
    assert "merge_raw_telemetry_csv" in code
    assert "persist_raw_telemetry" in code
    assert "class VAAET" not in code


def test_training_uses_shared_feature_contracts() -> None:
    code = _code(NOTEBOOKS["training"])
    assert "FEATURE_COLS" in code
    assert "from vaaet.features.engineering import engineer_features" in code
    assert "from vaaet.features.labeling import assign_traffic_state" in code
    assert "from vaaet.data.database import" in code
    assert "def engineer_features(" not in code
    assert "def assign_traffic_state(" not in code


def test_inference_uses_shared_analysis_and_validates_bundle() -> None:
    code = _code(NOTEBOOKS["inference"])
    assert "from vaaet.vision.analysis import TrafficStatePrediction, analyze_video" in code
    assert "validate_manifest(_model_dir_abs)" in code
    assert "from sqlalchemy import text as sa_text" in code
    assert "def estimate_speed(" not in code
    assert "def generate_annotated_video(" not in code
