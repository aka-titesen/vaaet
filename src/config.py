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


# Perception Pipeline Constants

# YOLO model variant selection by video duration (seconds)
YOLO_MODEL_VARIANTS: dict[str, dict[str, int | str]] = MappingProxyType(
    {  # type: ignore[assignment]
        "yolo11n": {"max_duration": 60, "label": "nano — clips < 1 min"},
        "yolo11s": {"max_duration": 180, "label": "small — clips 1-3 min"},
        "yolo11m": {"max_duration": 600, "label": "medium — clips 3-10 min"},
        "yolo11l": {"max_duration": 1800, "label": "large — clips 10-30 min"},
        "yolo11x": {"max_duration": 99999, "label": "xlarge — clips > 30 min"},
    }
)

# Default YOLO inference parameters
YOLO_CONFIDENCE: float = 0.5
YOLO_NMS_IOU: float = 0.4


# Tracker Constants

TRACKER_MAX_DISTANCE: float = 100.0  # Maximum Euclidean px for matching
TRACKER_MAX_LOST: int = 60  # Frames before track removal
TRACKER_HISTORY_MAXLEN: int = 50  # Centroid history deque length


# Optical Flow Constants

OPTICAL_FLOW_GRID_STEP: int = 40  # Pixel grid spacing for feature points
OPTICAL_FLOW_WIN_SIZE: tuple[int, int] = (21, 21)  # Lucas-Kanade window
OPTICAL_FLOW_MAX_LEVEL: int = 3  # Pyramid levels
OPTICAL_FLOW_RUNNING_MEAN: int = 30  # Frames for motion smoothing


# Speed Estimation Constants

PIXELS_PER_METER: float = 12.0  # Bridge-camera calibration factor

# Perspective correction zones (fraction of frame height)
PERSPECTIVE_ZONES: dict[str, dict[str, float]] = MappingProxyType(
    {  # type: ignore[assignment]
        "near": {"threshold": 0.66, "factor": 1.8},
        "mid": {"threshold": 0.33, "factor": 1.0},
        "far": {"threshold": 0.0, "factor": 0.6},
    }
)

# MLP smoother fusion weight: final = PHYSICS_WEIGHT * physics + MLP_WEIGHT * mlp
SPEED_PHYSICS_WEIGHT: float = 0.70
SPEED_MLP_WEIGHT: float = 0.30
SPEED_MLP_VALID_RANGE: tuple[float, float] = (5.0, 100.0)  # MLP plausibility

# Minimum track length (frames) before speed estimation is reliable
SPEED_MIN_TRACK_LENGTH: int = 20

# Per-vehicle-type speed limits (km/h) for plausibility filtering
SPEED_LIMITS_PER_TYPE: dict[str, tuple[float, float]] = MappingProxyType(
    {  # type: ignore[assignment]
        "car": (2.0, 120.0),
        "truck": (2.0, 90.0),
        "bus": (2.0, 80.0),
        "motorcycle": (2.0, 130.0),
        "bicycle": (2.0, 40.0),
    }
)


# Stationary Detection (AND-conjunction — see AGENTS.md)

STATIONARY_TOTAL_DISP_MAX: float = 5.0  # Total displacement in pixels
STATIONARY_MAX_SEGMENT_MAX: float = 3.0  # Max single-frame displacement
STATIONARY_STD_MAX: float = 2.5  # Std-dev of displacements
STATIONARY_AVG_FRAME_MAX: float = 0.3  # Average per-frame displacement
STATIONARY_MAX_FRAME_MAX: float = 1.5  # Max per-frame displacement


# Video I/O

# Strict filename format: bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4
VIDEO_FILENAME_PATTERN: str = (
    r"^bridge_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_to_\d{2}-\d{2}-\d{2}\.mp4$"
)
