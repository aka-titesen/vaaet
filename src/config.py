"""Centralized configuration for the VAAET intelligence layer.

This module is the single source of truth for paths, constants, and database
connection parameters used by both the data-preparation notebook and the
production traffic-analyzer notebook.
"""

from __future__ import annotations

import os
from types import MappingProxyType


# Reproducibility

RANDOM_SEED: int = 42


# Traffic state definitions

STATE_LABELS: dict[int, str] = MappingProxyType(
    {  # type: ignore[assignment]
        0: "Normal",
        1: "Reduced",
        2: "Congested",
        3: "Accident",
    }
)

# Number of traffic-state classes
N_STATES: int = len(STATE_LABELS)


# Feature columns (14) — canonical order used by scaler and model

FEATURE_COLS: list[str] = [
    "avg_speed",
    "total_vehicles",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "heavy_vehicle_ratio",
    "delta_speed",
    "delta_count",
    "transition_flag",
    "speed_variance",
    "hour_of_day",
    "weather_condition",
]


# Auto-labeling thresholds (Ask before modifying — see AGENTS.md)

LABELING_THRESHOLDS: dict[str, float | int] = MappingProxyType(
    {  # type: ignore[assignment]
        "accident_speed_max": 2,  # km/h — near zero
        "accident_delta_min": -20,  # km/h — sudden braking
        "accident_persistence": 3,  # consecutive records
        "congested_speed_max": 5,  # km/h
        "congested_vehicles_min": 25,  # vehicles / min
        "congested_persistence": 2,  # consecutive records
        "reduced_speed_min": 5,  # km/h
        "reduced_speed_max": 40,  # km/h
        "reduced_vehicles_min": 15,  # vehicles / min
        "reduced_vehicles_max": 25,  # vehicles / min
        "transition_delta_speed": 10,  # abs km/h change
        "transition_delta_count": 5,  # abs vehicle count change
        "rolling_window": 5,  # minutes for speed_variance
    }
)


# Artifact paths (relative to repository root)

MODEL_DIR: str = os.path.join("models", "intelligence")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
DATA_RAW_DIR: str = os.path.join("data", "raw")

MODEL_PATH: str = os.path.join(MODEL_DIR, "traffic_classifier.keras")
SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.joblib")
LABEL_MAP_PATH: str = os.path.join(MODEL_DIR, "label_mapping.joblib")


# Database

DB_ENV_VARS: tuple[str, ...] = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)

DEFAULT_DB_PORT: str = "5432"

# Model versioning
MODEL_VERSION: str = "mlp-v1.0"


# Bridge domain context

BRIDGE_CONFIG: dict[str, float | str] = MappingProxyType(
    {  # type: ignore[assignment]
        "name": "General Manuel Belgrano",
        "length_m": 1700,
        "road_width_m": 8.3,
        "camera_height_m": 60,
    }
)

VEHICLE_TYPES: tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle")

# Speed plausibility filter [min, max] in km/h
SPEED_RANGE: tuple[float, float] = (2.0, 120.0)
