from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ML_ROOT.parent
REPO_ROOT = ML_ROOT
SCRIPTS_DIR = ML_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ACTIVE_NOTEBOOKS = [
    NOTEBOOKS_DIR / "data-collection" / "collect_traffic_telemetry.ipynb",
    NOTEBOOKS_DIR / "training" / "train_traffic_state_classifier.ipynb",
    NOTEBOOKS_DIR / "inference" / "analyze_traffic_video.ipynb",
]
ALLOWED_SCRIPT_FILES = {
    "README.md",
    "audit-postgres-database.py",
    "convert-postgres-backup.py",
    "evaluate-telemetry-exports.py",
    "export-training-dataset.py",
    "setup-dvc.sh",
}


def test_training_and_ingestion_modules_import_without_cycles() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import vaaet_ml.data.ingestion; import vaaet_ml.training.partitions; "
            "import vaaet.inference.traffic_state",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


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
        for path in WORKSPACE_ROOT.joinpath("docs").rglob("*.md")
        if "architecture/decisions" not in path.relative_to(WORKSPACE_ROOT).as_posix()
    ]
    active_files = [
        ML_ROOT / "AGENTS.md",
        ML_ROOT / "CONTRIBUTING.md",
        ML_ROOT / "README.md",
        ML_ROOT / "llms.txt",
        WORKSPACE_ROOT / "AGENTS.md",
        WORKSPACE_ROOT / "CONTRIBUTING.md",
        WORKSPACE_ROOT / "README.md",
        WORKSPACE_ROOT / "SECURITY.md",
        WORKSPACE_ROOT / "SUPPORT.md",
        *REPO_ROOT.joinpath("src").rglob("*.py"),
        *REPO_ROOT.joinpath("scripts").rglob("*.py"),
        *ACTIVE_NOTEBOOKS,
        *active_docs,
        WORKSPACE_ROOT / "docs/architecture/decisions/0013-on-demand-data-collection-workflow.md",
        WORKSPACE_ROOT / "docs/architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md",
        WORKSPACE_ROOT / "docs/architecture/decisions/0015-postgresql-namespaces-security-and-hitl.md",
        WORKSPACE_ROOT / "docs/architecture/decisions/0016-postgresql-hardening-and-pipeline-runs.md",
        WORKSPACE_ROOT / "docs/architecture/decisions/0020-single-git-monorepo-and-application-boundary.md",
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
        assert 'WORKSPACE_DIR = Path("/content/vaaet")' in code
        assert 'ML_ROOT = WORKSPACE_DIR / "vaaet-ml"' in code
        assert "REPO_ROOT" in code
        assert code.count('"-e"') == 2
    diagram = (
        WORKSPACE_ROOT / "docs" / "architecture" / "diagrams" / "intelligence-pipeline.md"
    ).read_text(encoding="utf-8")
    assert "14 features" not in diagram
    assert "19 features" in diagram


def test_notebooks_do_not_own_database_schema_or_legacy_credentials() -> None:
    for notebook_path in ACTIVE_NOTEBOOKS:
        code = _concatenate_code_cells(notebook_path)
        assert "CREATE TABLE" not in code.upper()
        assert "ALTER TABLE" not in code.upper()
        assert "getpass(" not in code
        assert "hydrate_db_environment_from_colab" not in code


def test_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]*\]\((?!https?://|mailto:)([^)]+)\)")
    broken: list[str] = []
    for markdown_path in WORKSPACE_ROOT.rglob("*.md"):
        relative = markdown_path.relative_to(WORKSPACE_ROOT).as_posix()
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
        "collect_traffic_telemetry.ipynb": ("vision", "database"),
        "train_traffic_state_classifier.ipynb": ("inference", "training,visualization,database"),
        "analyze_traffic_video.ipynb": ("vision,inference", "visualization,database"),
    }
    for notebook in ACTIVE_NOTEBOOKS:
        code = _concatenate_code_cells(notebook)
        core_extras, ml_extras = expected[notebook.name]
        assert f'CORE_REQUIREMENT = f"{{CORE_ROOT}}[{core_extras}]"' in code
        assert f'ML_REQUIREMENT = f"{{ML_ROOT}}[{ml_extras}]"' in code


def test_python_313_is_declared_and_exercised_by_ci() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (WORKSPACE_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.14"' in pyproject
    assert '"Programming Language :: Python :: 3.13"' in pyproject
    assert 'python_version >= \'3.13\'' in pyproject
    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in workflow


def test_active_code_uses_semantic_telemetry_contract_names() -> None:
    active_python = [*REPO_ROOT.joinpath("src").rglob("*.py"), *REPO_ROOT.joinpath("tests").rglob("*.py")]
    forbidden = ("RAW_TELEMETRY_" + "V2_COLUMNS", "MODERN_" + "TELEMETRY_COLUMNS")
    for path in active_python:
        content = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in content, f"{path.relative_to(REPO_ROOT)} contains {name}"


def test_active_database_queries_do_not_select_star() -> None:
    for path in REPO_ROOT.joinpath("src", "vaaet_ml", "data").glob("*.py"):
        assert "SELECT *" not in path.read_text(encoding="utf-8").upper()


def test_monorepo_keeps_single_shared_workspace_roots() -> None:
    assert (WORKSPACE_ROOT / ".git").is_dir()
    assert (WORKSPACE_ROOT / ".dvc").is_dir()
    assert (WORKSPACE_ROOT / "docs").is_dir()
    assert (WORKSPACE_ROOT / "vaaet-app" / "README.md").is_file()
    assert (WORKSPACE_ROOT / "vaaet-core" / "pyproject.toml").is_file()
    assert (WORKSPACE_ROOT / "vaaet-core" / "src" / "vaaet").is_dir()
    assert (ML_ROOT / "pyproject.toml").is_file()
    assert (ML_ROOT / "src" / "vaaet_ml").is_dir()
    assert (ML_ROOT / "artifacts" / "traffic-state").is_dir()
