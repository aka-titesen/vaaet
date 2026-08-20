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
        "training": ("joblib", "numpy", "pandas", "psycopg2", "sqlalchemy", "tensorflow"),
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
    assert "if result.telemetry.empty:" in code
    assert "if not result.telemetry.empty and RAW_CSV.is_file():" in code
    assert "PostgreSQL persistence skipped" in code


def test_notebooks_handle_clips_without_complete_minutes() -> None:
    collection = _code(NOTEBOOKS["collection"])
    inference = _code(NOTEBOOKS["inference"])

    assert "minimum required: 60.0s" in collection
    assert "minimum required: 60.0s" in inference
    assert "df_classified = None" in inference
    assert "if df_classified.empty:" in inference
    assert "at least two consecutive complete 60-second windows" in inference
    assert inference.index("if df_classified.empty:") < inference.index(
        'df_classified["traffic_state"].unique()'
    )
    assert "INFERENCE_PIPELINE_RUN_ID = None" in inference
    assert "Feature engineering, classification, persistence, and HITL review were skipped" in inference
    assert "INFERENCE_PIPELINE_RUN_ID is not None" in inference
    assert "HITL review skipped" in inference


def test_training_uses_shared_feature_contracts() -> None:
    code = _code(NOTEBOOKS["training"])
    assert "FEATURE_COLS" in code
    assert "from vaaet.features.engineering import engineer_features" in code
    assert "from vaaet.features.labeling import assign_stable_traffic_state" in code
    assert "from vaaet.data.database import" in code
    assert "def engineer_features(" not in code
    assert "def assign_traffic_state(" not in code
    assert "from vaaet.training.partitions import build_training_partitions" in code
    assert "build_training_partitions(" in code
    assert "validation_data=(X_validation, y_validation)" in code
    assert "validation_split" not in code
    assert "Dense(n_classes, activation=\"softmax\")" in code
    assert "N_MODEL_STATES" in code
    assert "fit_temperature" in code
    assert "production_eligible" in code
    assert "SMOTE(" not in code
    assert "TrainingIngestionPlan(" in code
    assert "TrainingMode.SEED_BOOTSTRAP" in code
    assert "TrainingMode.HITL_RETRAINING" in code
    assert "SeedDatasetPackageSource" in code
    assert "HUMAN_HOLDOUT_FROZEN = False" in code
    assert "HumanHoldoutAction.REUSE_OR_CREATE" in code
    assert "HumanHoldoutAction.CREATE_NEW_VERSION" in code
    assert "HUMAN_HOLDOUT_UPDATE_REASON" in code
    assert "resolve_human_holdout(" in code
    assert "frozen_holdout=human_holdout_snapshot" in code
    assert "human_holdout_snapshot.descriptor" in code
    assert "/content/drive/MyDrive/vaaet-ml/data/holdouts" in code
    assert "no ephemeral fallback is allowed" in code
    assert "compose_supervised_dataset(" in code
    assert "VersionedSeedStore" in code
    assert "DatasetArtifactAction.REUSE_OR_CREATE" in code
    assert "DatasetArtifactAction.CREATE_NEW_VERSION" in code
    assert "HitlCatalogSource(HITL_CATALOG_PATH, CatalogSelection.ALL_ACTIVE)" in code
    assert "create_training_input_lock(" in code
    assert "training_input_lock=training_input_lock.descriptor" in code
    assert "data/processed/vaaet-seed-bootstrap-v1.zip" not in code
    assert "data/raw/vaaet-training-dataset-v1.zip" not in code
    assert 'feedback_policy=FeedbackPolicy.VALIDATED_ONLY' in code
    assert "USE_HUMAN_VALIDATED_FEEDBACK" not in code
    assert "persist_traffic_analysis" not in code


def test_training_prepares_postgres_backup_reader_in_colab() -> None:
    code = _code(NOTEBOOKS["training"])
    assert "ENABLE_DATA_UPLOAD = True" in code
    condition = 'TRAINING_MODE is TrainingMode.SEED_BOOTSTRAP and os.path.exists(_backup_dest) and not os.path.exists(_csv_dest)'
    assert condition in code
    assert 'Path("/usr/lib/postgresql/17/bin/pg_restore")' in code
    assert '["apt-get", "update", "-qq"]' in code
    assert '"postgresql-client-17"' in code
    assert "https://apt.postgresql.org/pub/repos/apt" in code
    assert "https://www.postgresql.org/media/keys/ACCC4CF8.asc" in code
    assert "PostgresBackupSource(BACKUP_PATH" in code
    assert "Path(PG_RESTORE_PATH) if PG_RESTORE_PATH else None" in code
    assert "Backup reader ready" in code
    assert "for fname in uploaded" in code
    assert "Immutable dataset root" in code
    assert "Detected backup table" in code
    assert "archive_table" in code
    assert "reader_version" in code
    assert "!apt-get" not in code
    assert "shell=True" not in code


def test_inference_finalizes_immutable_hitl_review_sessions() -> None:
    code = _code(NOTEBOOKS["inference"])
    assert "finalize_review_session" in code
    assert "def finalize_current_review" in code
    assert "REVIEW_VALIDATIONS.append" in code
    assert "/content/drive/MyDrive/vaaet-ml/data/hitl-reviews" in code
    assert "result.sync_status" in code
    assert "export_completed_offline_review" not in code


def test_inference_centralizes_and_documents_supported_workflow_configuration() -> None:
    notebook = json.loads(NOTEBOOKS["inference"].read_text(encoding="utf-8"))
    code = _code(NOTEBOOKS["inference"])
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assignments = {
        "ALLOW_PILOT_BUNDLE =": 1,
        "ALLOW_EXPERIMENTAL_BUNDLE =": 1,
        "PERSIST_TO_DATABASE =": 1,
        "ENABLE_HUMAN_REVIEW =": 1,
        "REVIEW_MODE =": 1,
        "DOWNLOAD_ANNOTATED_VIDEO =": 1,
        "SHOW_DASHBOARD =": 1,
        "HUD_DEBUG =": 1,
    }
    for assignment, expected_count in assignments.items():
        assert code.count(assignment) == expected_count

    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Workflow configuration — edit only this cell" in "".join(cell.get("source", []))
    )
    setup_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Environment setup — run once per Colab runtime" in "".join(cell.get("source", []))
    )
    assert config_index < setup_index
    assert 'if REVIEW_MODE not in {"priority", "all"}:' in code
    assert 'DEPLOYMENT_STAGE == "candidate" and PERSIST_TO_DATABASE' in code
    assert "Candidate bundles are offline-only" in code
    assert "if IN_COLAB and DOWNLOAD_ANNOTATED_VIDEO" in code
    assert "if not SHOW_DASHBOARD" in code
    assert "HudConfig(debug=HUD_DEBUG)" in code

    for heading in (
        "Inferencia piloto rápida",
        "Piloto con HITL portable",
        "Persistencia operacional sin revisión",
        "PostgreSQL con revisión prioritaria",
        "Revisión completa de un clip",
        "Candidato experimental",
        "Bundle aprobado para producción",
        "Clip corto o con un solo minuto completo",
    ):
        assert heading in markdown
    assert "try:\n    if df_telemetry" not in markdown


def test_collection_centralizes_safe_workflow_configuration() -> None:
    notebook = json.loads(NOTEBOOKS["collection"].read_text(encoding="utf-8"))
    code = _code(NOTEBOOKS["collection"])

    assert code.count("PERSIST_TO_DATABASE =") == 1
    assert code.count("HUD_DEBUG =") == 1
    assert "PERSIST_TO_DATABASE = False" in code
    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Workflow configuration — edit only this cell"
        in "".join(cell.get("source", []))
    )
    setup_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Environment setup — run once per Colab runtime"
        in "".join(cell.get("source", []))
    )
    assert config_index < setup_index


def test_training_resolves_postgres_only_after_explicit_opt_in() -> None:
    notebook = json.loads(NOTEBOOKS["training"].read_text(encoding="utf-8"))
    code = _code(NOTEBOOKS["training"])

    assert code.count("ENABLE_POSTGRES_INGESTION =") == 1
    assert "ENABLE_POSTGRES_INGESTION = False" in code
    assert code.count(
        "get_optional_database_settings(DatabaseProfile.TRAINING)"
    ) == 1
    guard_index = code.index("if ENABLE_POSTGRES_INGESTION:")
    settings_index = code.index(
        "get_optional_database_settings(DatabaseProfile.TRAINING)"
    )
    assert guard_index < settings_index
    assert "PostgreSQL ingestion is enabled, but the read-only training profile" in code
    assert "PERSIST_TO_DATABASE" not in code

    setup_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Environment setup — run once per Colab runtime"
        in "".join(cell.get("source", []))
    )
    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Training workflow configuration — edit only this cell"
        in "".join(cell.get("source", []))
    )
    upload_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "# Cell 1b — Data Upload (Colab only)" in "".join(cell.get("source", []))
    )
    assert setup_index < config_index < upload_index


def test_training_augmentation_handles_raw_and_feedback_inputs() -> None:
    code = _code(NOTEBOOKS["training"])
    guard = 'if "df_raw" not in globals() or not isinstance(df_raw, pd.DataFrame):'
    assert guard in code
    assert code.index(guard) < code.index("_n_before = len(df_raw)")
    assert "Synthetic raw augmentation skipped outside the one-time seed bootstrap" in code
    assert "from vaaet.data.timestamps import normalize_timestamp_series" in code
    assert "Canonical timestamp timezone" in code


def test_training_documents_actual_synthetic_record_count() -> None:
    notebook = NOTEBOOKS["training"].read_text(encoding="utf-8")
    assert "200 records total" in notebook
    assert "100 records total" not in notebook


def test_training_compares_conservative_balance_candidates() -> None:
    code = _code(NOTEBOOKS["training"])
    assert "build_balance_candidates(" in code
    assert "BalanceStrategy.CLASS_WEIGHTS" not in code
    assert "BalanceStrategy.SYNTHETIC_CONGESTION" in code
    assert "validation_false_congested_rate" in code
    assert "selection_score" in code
    assert "SELECTED_BALANCE_STRATEGY" in code
    assert "expired proxy-memory rows before scaling" in code
    assert "labels=[0, 1, 2]" in code


def test_training_applies_same_legacy_policy_as_inference() -> None:
    training = _code(NOTEBOOKS["training"])
    inference = _code(NOTEBOOKS["inference"])
    assert "ModelInputPolicy.LEGACY_V1_BOOTSTRAP" in training
    assert "apply_model_input_policy(" in training
    assert "input_policy=MODEL_INPUT_POLICY" in training
    assert 'MODEL_INPUT_POLICY = manifest["training_lifecycle"]["input_policy"]' in inference
    assert "input_policy=MODEL_INPUT_POLICY" in inference


def test_inference_uses_shared_analysis_and_validates_bundle() -> None:
    code = _code(NOTEBOOKS["inference"])
    assert "from vaaet.vision.analysis import TrafficStatePrediction, analyze_video" in code
    assert "validate_manifest(_model_dir_abs)" in code
    assert "from sqlalchemy import text as sa_text" not in code
    assert "load_review_queue" in code
    assert "build_review_widget" in code
    assert "finalize_current_review" in code
    assert "DatabaseProfile.REVIEW" in code
    assert "ENABLE_HUMAN_REVIEW = False" in code
    assert "PERSIST_TO_DATABASE = False" in code
    assert "validation_split" not in code
    assert "SMOTE" not in code
    assert "def estimate_speed(" not in code
    assert "def generate_annotated_video(" not in code
    assert 'decision_policy=manifest["decision_policy"]' in code
    assert "ALLOW_EXPERIMENTAL_BUNDLE" in code
    assert "ALLOW_PILOT_BUNDLE" in code
    assert "DEPLOYMENT_STAGE" in code
    assert 'model_version=manifest["model_version"]' in code
    assert "retrain_with_feedback" not in code
    assert "model.output_shape[-1]" in code
    assert "dict(label_mapping) != dict(STATE_LABELS)" in code


def test_annotated_video_workflows_default_to_public_shared_hud() -> None:
    collection = _code(NOTEBOOKS["collection"])
    inference = _code(NOTEBOOKS["inference"])
    for code in (collection, inference):
        assert code.count("HUD_DEBUG = False") == 1
        assert "from vaaet.vision.hud import HudConfig" in code
        assert "hud_config=HudConfig(debug=HUD_DEBUG)" in code
    assert 'incident_candidate=bool(latest.get("accident_rule_triggered", False))' in inference


def test_notebooks_use_profile_specific_database_api() -> None:
    collection = _code(NOTEBOOKS["collection"])
    inference = _code(NOTEBOOKS["inference"])
    training = _code(NOTEBOOKS["training"])

    assert "DatabaseProfile.COLLECTION" in collection
    assert "DatabaseProfile.INFERENCE" in inference
    assert "DatabaseProfile.REVIEW" in inference
    assert "DatabaseProfile.TRAINING" in training
    for code in (collection, inference, training):
        assert "hydrate_db_environment_from_colab" not in code
        assert 'os.environ["DB_PASSWORD"]' not in code
        assert "getpass(" not in code


def test_all_workflows_record_redacted_pipeline_runs() -> None:
    for workflow, path in NOTEBOOKS.items():
        code = _code(path)
        assert "PipelineRunMetadata" in code, workflow
        assert "PipelineWorkflow" in code, workflow
        assert "pipeline_run(" in code, workflow
        assert "data/processed/pipeline-runs" in code, workflow
