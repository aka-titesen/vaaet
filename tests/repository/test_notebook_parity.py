"""Verify that all notebooks orchestrate shared package APIs."""

from __future__ import annotations

import json
from pathlib import Path

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
