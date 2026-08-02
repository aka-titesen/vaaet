from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ACTIVE_NOTEBOOKS = [
    NOTEBOOKS_DIR / "data-collection" / "collect_traffic_telemetry.ipynb",
    NOTEBOOKS_DIR / "training" / "train_traffic_state_classifier.ipynb",
    NOTEBOOKS_DIR / "inference" / "analyze_traffic_video.ipynb",
]
ALLOWED_SCRIPT_FILES = {
    "README.md",
    "convert-postgres-backup.py",
    "evaluate-telemetry-exports.py",
    "setup-dvc.sh",
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
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"{notebook_path}:cell-{index}")


def test_active_notebook_folder_has_no_backups() -> None:
    backups = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in NOTEBOOKS_DIR.rglob("*.ipynb.bak")
    )
    assert backups == []


def test_scripts_directory_only_contains_supported_utilities() -> None:
    files = {path.name for path in SCRIPTS_DIR.iterdir() if path.is_file()}
    assert files == ALLOWED_SCRIPT_FILES


def test_active_sources_do_not_use_legacy_paths_or_import_hacks() -> None:
    forbidden = (
        "archive/bootstrap-v1",
        "notebooks/01_data_prep",
        "notebooks/02_production",
        "docs/adr/",
        "data/samples",
        "src/perception",
        "from src.",
        "import src.",
        "sys.path.insert",
        "install_if_missing",
        "vaaet-ml.git",
    )
    active_docs = [
        path
        for path in REPO_ROOT.joinpath("docs").rglob("*.md")
        if "architecture/decisions" not in path.relative_to(REPO_ROOT).as_posix()
    ]
    active_files = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "SUPPORT.md",
        REPO_ROOT / "llms.txt",
        *REPO_ROOT.joinpath("src").rglob("*.py"),
        *REPO_ROOT.joinpath("scripts").rglob("*.py"),
        *ACTIVE_NOTEBOOKS,
        *active_docs,
        REPO_ROOT / "docs/architecture/decisions/0012-ml-web-boundary-and-artifact-contract.md",
        REPO_ROOT / "docs/architecture/decisions/0013-on-demand-data-collection-workflow.md",
    ]
    for path in active_files:
        content = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in content, f"{path.relative_to(REPO_ROOT)} contains {value!r}"
    inference = ACTIVE_NOTEBOOKS[-1].read_text(encoding="utf-8")
    assert "os.path.join(_root" not in inference


def test_active_notebooks_use_canonical_feature_count_and_repository_root() -> None:
    for notebook_path in ACTIVE_NOTEBOOKS:
        content = notebook_path.read_text(encoding="utf-8")
        code = _concatenate_code_cells(notebook_path)
        assert "14 features" not in content
        assert "vaaet-ml.git" not in content
        assert 'REPO_DIR = Path("/content/vaaet")' in code
        assert "REPO_ROOT" in code
        assert code.count('"-e"') == 1
    diagram = (
        REPO_ROOT / "docs" / "architecture" / "diagrams" / "intelligence-pipeline.md"
    ).read_text(encoding="utf-8")
    assert "14 features" not in diagram
    assert "19 features" in diagram


def test_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]*\]\((?!https?://|mailto:)([^)]+)\)")
    broken: list[str] = []
    for markdown_path in REPO_ROOT.rglob("*.md"):
        relative = markdown_path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("plantillas_docs/"):
            continue
        for target_text in link_pattern.findall(markdown_path.read_text(encoding="utf-8")):
            clean_target = target_text.split("#", 1)[0].split("?", 1)[0]
            if clean_target and not (markdown_path.parent / clean_target).resolve().exists():
                broken.append(f"{relative} -> {clean_target}")
    assert broken == []


def test_removed_directories_are_absent() -> None:
    removed = (
        "archive",
        "models",
        "notebooks/01_data_prep",
        "notebooks/02_production",
        "docs/adr",
        "docs/diagrams",
        "docs/KPIs",
        "src/perception",
        "data/samples",
        "data/interim",
    )
    assert [path for path in removed if (REPO_ROOT / path).exists()] == []


def test_notebook_extras_match_workflows() -> None:
    expected = {
        "collect_traffic_telemetry.ipynb": "vision,database",
        "train_traffic_state_classifier.ipynb": "training,visualization,database",
        "analyze_traffic_video.ipynb": "vision,training,visualization,database",
    }
    for notebook in ACTIVE_NOTEBOOKS:
        code = _concatenate_code_cells(notebook)
        assert f"[{expected[notebook.name]}]" in code
