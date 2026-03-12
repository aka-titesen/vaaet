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


# Auto-labeling thresholds — calibrated to Belgrano Bridge real data
# (P25 speed ≈ 7.78 km/h, P75 ≈ 18.52; median vehicles ≈ 3, P75 ≈ 6)
# Recalibrated 2026-03-11 from generic textbook values to bridge-specific
# percentiles so that all 4 classes appear in the ~2 000-record dataset.

LABELING_THRESHOLDS: dict[str, float | int] = MappingProxyType(
    {  # type: ignore[assignment]
        "accident_speed_max": 2,  # km/h — near zero (unchanged)
        "accident_delta_min": -15,  # km/h — sudden braking (was -20)
        "accident_cumulative_delta_min": -18,  # km/h — multi-step braking window
        "accident_persistence": 2,  # consecutive records (was 3)
        "congested_speed_max": 7,  # km/h — below P25 of bridge data (was 5)
        "congested_vehicles_min": 8,  # ≈P85 of bridge volume (was 25)
        "congested_persistence": 2,  # consecutive records (unchanged)
        "reduced_speed_min": 7,  # km/h — = congested_speed_max (was 5)
        "reduced_speed_max": 25,  # km/h — ≈P80 of bridge speeds (was 40)
        "reduced_vehicles_min": 5,  # ≈P65 of bridge volume (was 15)
        "reduced_vehicles_max": 12,  # ≈P95 of bridge volume (was 25)
        "transition_delta_speed": 8,  # abs km/h change (was 10)
        "transition_delta_count": 3,  # abs vehicle count change (was 5)
        "rolling_window": 5,  # minutes for speed_variance (unchanged)
    }
)


# Artifact paths (relative to repository root)

MODEL_DIR: str = os.path.join("models", "intelligence")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
DATA_RAW_DIR: str = os.path.join("data", "raw")

MODEL_PATH: str = os.path.join(MODEL_DIR, "traffic_classifier.keras")
SCALER_PATH: str = os.path.join(MODEL_DIR, "feature_scaler.joblib")
LABEL_MAP_PATH: str = os.path.join(MODEL_DIR, "label_mapping.joblib")

# Google Drive artifact bridge (Colab persistence across sessions)
# Mount point: /content/drive/ — artefacts copied here after M1 training
# so M2 can load them even after a runtime reset.
DRIVE_ARTIFACT_DIR: str = os.path.join("MyDrive", "vaaet", "models", "intelligence")


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
# Logic: shorter clips can afford heavier models (higher accuracy, fewer frames).
# Longer clips require lighter models to finish in reasonable time.
# Matches ADR-002 adaptive selection strategy from legacy pipeline.
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

TRACKER_MAX_DISTANCE: float = 100.0  # Maximum Euclidean px for matching
TRACKER_MAX_LOST: int = 60  # Frames before track removal
TRACKER_HISTORY_MAXLEN: int = 50  # Centroid history deque length


# Optical Flow Constants

OPTICAL_FLOW_GRID_STEP: int = 40  # Pixel grid spacing for feature points
OPTICAL_FLOW_BORDER_MARGIN: int = 20  # Skip low-quality border points
OPTICAL_FLOW_WIN_SIZE: tuple[int, int] = (21, 21)  # Lucas-Kanade window
OPTICAL_FLOW_MAX_LEVEL: int = 3  # Pyramid levels
OPTICAL_FLOW_RUNNING_MEAN: int = 30  # Frames for motion smoothing
OPTICAL_FLOW_MIN_TRACKING_RATIO: float = 0.35  # Reject low-confidence flow


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

# Linear blend band around perspective thresholds (fraction of frame height).
# Prevents abrupt speed changes when a vehicle crosses near/mid/far zone borders.
PERSPECTIVE_BLEND_BAND: float = 0.05

# MLP smoother fusion weight: final = PHYSICS_WEIGHT * physics + MLP_WEIGHT * mlp
SPEED_PHYSICS_WEIGHT: float = 0.70
SPEED_MLP_WEIGHT: float = 0.30
SPEED_MLP_VALID_RANGE: tuple[float, float] = (5.0, 100.0)  # MLP plausibility
SPEED_RECOVERY_SKIP_GAP: int = 1  # Skip speed on the first frame after recovery
SPEED_ROBUST_TRIM_RATIO: float = 0.15  # Trim minute-level outliers when possible
SPEED_ROBUST_OUTLIER_SIGMA: float = 3.5  # Modified-z threshold for outliers

# Minimum track length (frames) before speed estimation is reliable
SPEED_MIN_TRACK_LENGTH: int = 8

# Rolling window (frames) for speed calculation (~1 s at 30 fps, matches legacy)
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


# Stationary Detection (AND-conjunction — see AGENTS.md)

STATIONARY_TOTAL_DISP_MAX: float = 5.0  # Total displacement in pixels
STATIONARY_MAX_SEGMENT_MAX: float = 3.0  # Max single-frame displacement
STATIONARY_STD_MAX: float = 2.5  # Std-dev of displacements
STATIONARY_AVG_FRAME_MAX: float = 0.3  # Average per-frame displacement
STATIONARY_MAX_FRAME_MAX: float = 1.5  # Max per-frame displacement
STATIONARY_ENTRY_FRAMES: int = 2  # Hysteresis: consecutive stationary votes to enter
STATIONARY_EXIT_FRAMES: int = 3  # Hysteresis: consecutive moving votes to exit
STATIONARY_EXIT_SPEED_MIN: float = 6.0  # Speed threshold to leave stationary state


# Video I/O

# Strict filename format: bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4
VIDEO_FILENAME_PATTERN: str = (
    r"^bridge_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_to_\d{2}-\d{2}-\d{2}\.mp4$"
)
