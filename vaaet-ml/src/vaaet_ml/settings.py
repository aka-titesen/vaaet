"""Configuración operativa exclusiva del laboratorio VAAET ML."""

from __future__ import annotations

import os

from vaaet.settings import *  # noqa: F403

RANDOM_SEED: int = 42
MODEL_DIR: str = os.path.join("artifacts", "traffic-state")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
DATA_RAW_DIR: str = os.path.join("data", "raw")
MODEL_PATH: str = os.path.join(MODEL_DIR, "traffic_classifier.keras")
SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.joblib")
LABEL_MAP_PATH: str = os.path.join(MODEL_DIR, "label_mapping.joblib")
DRIVE_ARTIFACT_DIR: str = os.path.join("MyDrive", "vaaet-ml", "artifacts", "traffic-state")
DB_ENV_VARS: tuple[str, ...] = ("VAAET_DB_HOST", "VAAET_DB_PORT", "VAAET_DB_NAME")
DEFAULT_DB_PORT: str = "5432"
DATABASE_SCHEMA_VERSION: str = "vaaet-db-v2"
DATABASE_SCHEMAS: tuple[str, ...] = (
    "vaaet_raw",
    "vaaet_ml",
    "vaaet_feedback",
    "vaaet_ops",
)
