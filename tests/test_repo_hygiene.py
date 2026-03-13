from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ACTIVE_NOTEBOOKS = [
    NOTEBOOKS_DIR / "01_data_prep" / "data_preparation.ipynb",
    NOTEBOOKS_DIR / "02_production" / "traffic_analyzer.ipynb",
]
ALLOWED_SCRIPT_FILES = {
    "README.md",
    "convert_backup.py",
    "evaluate_real_clips.py",
}


def _concatenate_code_cells(notebook_path: Path) -> str:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_active_notebooks_compile() -> None:
    for notebook_path in ACTIVE_NOTEBOOKS:
        code = _concatenate_code_cells(notebook_path)
        compile(code, str(notebook_path), "exec")


def test_active_notebook_folder_has_no_backups() -> None:
    backups = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in NOTEBOOKS_DIR.rglob("*.ipynb.bak")
    )
    assert backups == []


def test_scripts_directory_only_contains_supported_utilities() -> None:
    files = {path.name for path in SCRIPTS_DIR.iterdir() if path.is_file()}
    assert files == ALLOWED_SCRIPT_FILES
