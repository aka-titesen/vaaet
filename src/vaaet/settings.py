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

# Temporal contract

CANONICAL_TIMEZONE: str = "UTC"
TRAFFIC_LOCAL_TIMEZONE: str = "America/Argentina/Buenos_Aires"


# Dataset provenance

DATA_ORIGIN_COL: str = "data_origin"
SYNTHETIC_SCENARIO_COL: str = "synthetic_scenario"
DATA_ORIGINS: tuple[str, ...] = ("real", "synthetic")
SYNTHETIC_SCENARIOS: tuple[str, ...] = ("observed", "accident", "congestion")

# Traffic state definitions

STATE_LABELS: dict[int, str] = MappingProxyType(
    {  # type: ignore[assignment]
        0: "Normal",
        1: "Reduced",
        2: "Congested",
        3: "Accident",
    }
)

# The public contract keeps four states, while the learned classifier models
# only the stable traffic-flow states. Accident is a human-confirmed outcome
# produced by the hierarchical decision policy, never a direct MLP output.
MODEL_STATE_LABELS: dict[int, str] = MappingProxyType(  # type: ignore[assignment]
    {key: STATE_LABELS[key] for key in (0, 1, 2)}
)

# Number of traffic-state classes
N_STATES: int = len(STATE_LABELS)
N_MODEL_STATES: int = len(MODEL_STATE_LABELS)


# Feature columns (19) — canonical order used by scaler and model

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
    "cumulative_delta_speed",
    "low_speed_persistence",
    "speed_measurement_quality",
    "near_zero_motion_ratio",
    "stationary_confirmed_ratio",
    "hour_of_day",
    "weather_condition",
]


# Auto-labeling thresholds — calibrated to Belgrano Bridge real data
# (P25 speed ≈ 7.78 km/h, P75 ≈ 18.52; median vehicles ≈ 3, P75 ≈ 6)
# Recalibrated 2026-03-11 from generic textbook values to bridge-specific
# percentiles so that all 4 classes appear in the ~2 000-record dataset.

LABELING_THRESHOLDS: dict[str, float | int] = MappingProxyType(
    {  # type: ignore[assignment]
        "accident_speed_max": 2,
        "accident_delta_min": -15,
        "accident_cumulative_delta_min": -18,
        "accident_persistence": 2,
        "congested_speed_max": 7,
        "congested_vehicles_min": 5,
        "congested_persistence": 2,
        "reduced_speed_min": 7,
        "reduced_speed_max": 25,
        "reduced_vehicles_min": 5,
        "reduced_vehicles_max": 12,
        "transition_delta_speed": 8,
        "transition_delta_count": 3,
        "rolling_window": 5,
    }
)

ACCIDENT_GATE_MIN_EVIDENCE_SCORE: float = 0.75
SPEED_MEASUREMENT_QUALITY_MIN: float = 0.45
NEAR_ZERO_RATIO_MIN: float = 0.20
STATIONARY_CONFIRMED_RATIO_MIN: float = 0.10
OPTICAL_FLOW_QUALITY_MIN: float = 0.35
INCIDENT_PERSISTENCE_MINUTES: int = 2
INCIDENT_RECOVERY_MINUTES: int = 3

# Conservative post-model policy. Concrete per-class thresholds are stored in
# bundle v2 so they can be selected on validation without changing the API.
DEFAULT_CLASS_THRESHOLDS: dict[int, float] = MappingProxyType(  # type: ignore[assignment]
    {0: 0.60, 1: 0.60, 2: 0.70}
)
DEFAULT_MIN_PROBABILITY_MARGIN: float = 0.10
WORSENING_PERSISTENCE_MINUTES: int = 2
RECOVERY_PERSISTENCE_MINUTES: int = 3
FEATURE_MAX_GAP_MINUTES: int = 2


# Artifact paths (relative to repository root)

MODEL_DIR: str = os.path.join("artifacts", "traffic-state")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
DATA_RAW_DIR: str = os.path.join("data", "raw")

MODEL_PATH: str = os.path.join(MODEL_DIR, "traffic_classifier.keras")
SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.joblib")
LABEL_MAP_PATH: str = os.path.join(MODEL_DIR, "label_mapping.joblib")

# Google Drive artifact bridge (Colab persistence across sessions)
# Mount point: /content/drive/ — artefacts copied here after M1 training
# so M2 can load them even after a runtime reset.
DRIVE_ARTIFACT_DIR: str = os.path.join("MyDrive", "vaaet-ml", "artifacts", "traffic-state")


# Database

DB_ENV_VARS: tuple[str, ...] = (
    "VAAET_DB_HOST",
    "VAAET_DB_PORT",
    "VAAET_DB_NAME",
)

DEFAULT_DB_PORT: str = "5432"
DATABASE_SCHEMA_VERSION: str = "vaaet-db-v2"
DATABASE_SCHEMAS: tuple[str, ...] = (
    "vaaet_raw",
    "vaaet_ml",
    "vaaet_feedback",
    "vaaet_ops",
)

# Model versioning
MODEL_VERSION: str = "mlp-v2.1"
TELEMETRY_SCHEMA_VERSION: str = "traffic-telemetry-v2"


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
# Logic: shorter clips can afford heavier models (higher accuracy, fewer frames).
# Longer clips require lighter models to finish in reasonable time.
# Implements the adaptive selection strategy in ADR-0002.
YOLO_MODEL_VARIANTS: dict[str, dict[str, int | str]] = MappingProxyType(
    {  # type: ignore[assignment]
        "yolo11x": {"max_duration": 300, "label": "xlarge — clips < 5 min"},
        "yolo11l": {"max_duration": 1800, "label": "large — clips 5-30 min"},
        "yolo11m": {"max_duration": 10800, "label": "medium — clips 30 min-3 h"},
        "yolo11s": {"max_duration": 43200, "label": "small — clips 3-12 h"},
        "yolo11n": {"max_duration": 99999, "label": "nano — clips > 12 h"},
    }
)

# Default YOLO inference parameters
YOLO_CONFIDENCE: float = 0.5
YOLO_NMS_IOU: float = 0.4


# Tracker Constants

TRACKER_MAX_DISTANCE: float = 100.0
TRACKER_MAX_LOST: int = 60
TRACKER_HISTORY_MAXLEN: int = 50


# Optical Flow Constants

OPTICAL_FLOW_GRID_STEP: int = 40
OPTICAL_FLOW_BORDER_MARGIN: int = 20
OPTICAL_FLOW_WIN_SIZE: tuple[int, int] = (21, 21)
OPTICAL_FLOW_MAX_LEVEL: int = 3
OPTICAL_FLOW_RUNNING_MEAN: int = 30
OPTICAL_FLOW_MIN_TRACKING_RATIO: float = 0.35


# Speed Estimation Constants

PIXELS_PER_METER: float = 12.0

# Perspective correction zones (fraction of frame height)
PERSPECTIVE_ZONES: dict[str, dict[str, float]] = MappingProxyType(
    {  # type: ignore[assignment]
        "near": {"threshold": 0.66, "factor": 1.8},
        "mid": {"threshold": 0.33, "factor": 1.0},
        "far": {"threshold": 0.0, "factor": 0.6},
    }
)

# Linear blend band around perspective thresholds (fraction of frame height).
# Prevents abrupt speed changes when a vehicle crosses near/mid/far zone borders.
PERSPECTIVE_BLEND_BAND: float = 0.05

# MLP smoother fusion weight: final = PHYSICS_WEIGHT * physics + MLP_WEIGHT * mlp
SPEED_PHYSICS_WEIGHT: float = 0.70
SPEED_MLP_WEIGHT: float = 0.30
SPEED_MLP_VALID_RANGE: tuple[float, float] = (5.0, 100.0)
SPEED_RECOVERY_SKIP_GAP: int = 1
SPEED_ROBUST_TRIM_RATIO: float = 0.15
SPEED_ROBUST_OUTLIER_SIGMA: float = 3.5

# Minimum track length (frames) before speed estimation is reliable
SPEED_MIN_TRACK_LENGTH: int = 8

# Rolling window for speed calculation (approximately 1 s at 30 FPS).
SPEED_ESTIMATION_WINDOW: int = 30

# Per-frame displacement noise floor (px). Displacements below this are
# zeroed to prevent tracking jitter from producing phantom speed.
SPEED_DISPLACEMENT_NOISE_FLOOR: float = 2.0

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


# Near-zero motion detection (broader than stationary)

NEAR_ZERO_TOTAL_DISP_MAX: float = 12.0
NEAR_ZERO_MAX_SEGMENT_MAX: float = 6.0
NEAR_ZERO_STD_MAX: float = 4.0
NEAR_ZERO_AVG_FRAME_MAX: float = 1.2
NEAR_ZERO_MAX_FRAME_MAX: float = 3.5


# Stationary Detection (AND-conjunction — see AGENTS.md)

STATIONARY_TOTAL_DISP_MAX: float = 5.0
STATIONARY_MAX_SEGMENT_MAX: float = 3.0
STATIONARY_STD_MAX: float = 2.5
STATIONARY_AVG_FRAME_MAX: float = 0.3
STATIONARY_MAX_FRAME_MAX: float = 1.5
STATIONARY_ENTRY_FRAMES: int = 2
STATIONARY_EXIT_FRAMES: int = 3
STATIONARY_EXIT_SPEED_MIN: float = 6.0


# Video I/O

# Strict filename format: bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4
VIDEO_FILENAME_PATTERN: str = (
    r"^bridge_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_to_\d{2}-\d{2}-\d{2}\.mp4$"
)
