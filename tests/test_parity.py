"""Parity tests — verify that notebooks import from src/ (no duplication).

These tests read the notebook JSON and confirm that key functions and
constants are imported, not redefined inline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_1_PATH = REPO_ROOT / "notebooks" / "01_data_prep" / "data_preparation.ipynb"
MODULE_2_PATH = REPO_ROOT / "notebooks" / "02_production" / "traffic_analyzer.ipynb"


def _get_all_code(notebook_path: Path) -> str:
    """Concatenate all code cells from a notebook into one string."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]
    return "\n".join(code_cells)


class TestModule1Parity:
    """Module 1 (data_preparation) must use shared src/ modules."""

    @pytest.fixture(autouse=True)
    def _load_code(self) -> None:
        self.code = _get_all_code(MODULE_1_PATH)

    def test_imports_feature_cols_from_config(self) -> None:
        # Match actual import statements (not comments) with multi-line support
        match = re.search(
            r"^from src\.config import \((.+?)\)",
            self.code,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, "No 'from src.config import (...)' found"
        assert "FEATURE_COLS" in match.group(1)

    def test_imports_state_labels_from_config(self) -> None:
        match = re.search(
            r"^from src\.config import \((.+?)\)",
            self.code,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None
        assert "STATE_LABELS" in match.group(1)

    def test_imports_engineer_features(self) -> None:
        assert "from src.features import engineer_features" in self.code

    def test_imports_assign_traffic_state(self) -> None:
        assert "from src.labeling import assign_traffic_state" in self.code

    def test_imports_db_functions(self) -> None:
        assert "from src.db import" in self.code

    def test_no_inline_engineer_features(self) -> None:
        """Should NOT redefine engineer_features() locally."""
        assert "def engineer_features(" not in self.code

    def test_no_inline_assign_traffic_state(self) -> None:
        """Should NOT redefine assign_traffic_state() locally."""
        assert "def assign_traffic_state(" not in self.code

    def test_no_inline_get_db_config(self) -> None:
        """Should NOT redefine get_db_config() locally."""
        assert "def get_db_config(" not in self.code

    def test_no_inline_load_telemetry(self) -> None:
        """Should NOT redefine load_telemetry() locally."""
        assert "def load_telemetry(" not in self.code

    def test_sys_path_insert(self) -> None:
        """Cell 0 must add the repo root to sys.path."""
        assert "sys.path.insert" in self.code


class TestModule2Parity:
    """Module 2 (traffic_analyzer) must use shared src/ modules."""

    @pytest.fixture(autouse=True)
    def _load_code(self) -> None:
        self.code = _get_all_code(MODULE_2_PATH)

    def test_imports_from_src_config(self) -> None:
        assert "from src.config import" in self.code

    def test_imports_from_src_features(self) -> None:
        assert "from src.features import engineer_features" in self.code

    def test_imports_from_src_labeling(self) -> None:
        assert "from src.labeling import assign_traffic_state" in self.code

    def test_sys_path_insert(self) -> None:
        assert "sys.path.insert" in self.code

    def test_imports_yolo_detector(self) -> None:
        assert "from src.perception.detector import" in self.code

    def test_imports_sort_tracker(self) -> None:
        assert "from src.perception.tracker import SORTTracker" in self.code

    def test_imports_speed_estimation(self) -> None:
        assert "from src.perception.speed import" in self.code

    def test_imports_optical_flow(self) -> None:
        assert (
            "from src.perception.optical_flow import OpticalFlowEstimator" in self.code
        )

    def test_imports_video_utils(self) -> None:
        assert "from src.video import" in self.code

    def test_imports_select_model_variant(self) -> None:
        assert "select_model_variant" in self.code

    def test_imports_is_stationary(self) -> None:
        assert "is_stationary" in self.code

    def test_imports_smoothed_speed_tracker(self) -> None:
        assert "SmoothedSpeedTracker" in self.code

    def test_no_inline_engineer_features(self) -> None:
        assert "def engineer_features(" not in self.code

    def test_no_inline_assign_traffic_state(self) -> None:
        assert "def assign_traffic_state(" not in self.code

    def test_no_inline_estimate_speed(self) -> None:
        assert "def estimate_speed(" not in self.code

    def test_no_inline_get_perspective_factor(self) -> None:
        assert "def get_perspective_factor(" not in self.code
