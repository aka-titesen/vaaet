# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contratos algorítmicos portables de VAAET Core.

El core define tiempo, telemetría, features, etiquetas y compatibilidad del
bundle. Las rutas, bases de datos, DVC, Drive y notebooks pertenecen al laboratorio.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# Contrato temporal

CANONICAL_TIMEZONE: Final[str] = "UTC"
TRAFFIC_LOCAL_TIMEZONE: Final[str] = "America/Argentina/Buenos_Aires"


# Procedencia de datasets

DATA_ORIGIN_COL: Final[str] = "data_origin"
SYNTHETIC_SCENARIO_COL: Final[str] = "synthetic_scenario"
DATA_ORIGINS: Final[tuple[str, ...]] = ("real", "synthetic")
SYNTHETIC_SCENARIOS: Final[tuple[str, ...]] = ("observed", "accident", "congestion")

# Estados de tránsito

STATE_LABELS: Final[Mapping[int, str]] = MappingProxyType(
    {
        0: "Normal",
        1: "Reduced",
        2: "Congested",
        3: "Accident",
    }
)

# El contrato público conserva cuatro estados, pero el MLP sólo aprende los
# tres estados estables. Accident exige confirmación humana.
MODEL_STATE_LABELS: Final[Mapping[int, str]] = MappingProxyType(
    {key: STATE_LABELS[key] for key in (0, 1, 2)}
)

N_STATES: Final[int] = len(STATE_LABELS)
N_MODEL_STATES: Final[int] = len(MODEL_STATE_LABELS)


# Orden canónico de las 19 features que comparten scaler y modelo.

FEATURE_COLS: Final[list[str]] = [
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


# Umbrales recalibrados el 2026-03-11 con percentiles del Puente Belgrano:
# velocidad P25 ≈ 7,78 km/h, P75 ≈ 18,52; mediana vehicular ≈ 3 y P75 ≈ 6.

LABELING_THRESHOLDS: Final[Mapping[str, float | int]] = MappingProxyType(
    {
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

# Los umbrales por clase se guardan en el bundle v2 para seleccionarlos sobre
# validación sin modificar la API.
DEFAULT_CLASS_THRESHOLDS: Final[Mapping[int, float]] = MappingProxyType(
    {0: 0.60, 1: 0.60, 2: 0.70}
)
DEFAULT_MIN_PROBABILITY_MARGIN: float = 0.10
WORSENING_PERSISTENCE_MINUTES: int = 2
RECOVERY_PERSISTENCE_MINUTES: int = 3
FEATURE_MAX_GAP_MINUTES: int = 2


# Versionado contractual
MODEL_VERSION: str = "mlp-v2.1"
TELEMETRY_SCHEMA_VERSION: str = "traffic-telemetry-v2"


# Contexto físico del puente

BRIDGE_CONFIG: Final[Mapping[str, float | str]] = MappingProxyType(
    {
        "name": "General Manuel Belgrano",
        "length_m": 1700,
        "road_width_m": 8.3,
        "camera_height_m": 60,
    }
)

VEHICLE_TYPES: Final[tuple[str, ...]] = ("car", "truck", "bus", "motorcycle", "bicycle")

SPEED_RANGE: tuple[float, float] = (2.0, 120.0)


# La duración del video selecciona una variante YOLO: los clips breves toleran
# modelos más pesados y los extensos priorizan finalizar en tiempo razonable.
YOLO_MODEL_VARIANTS: Final[Mapping[str, Mapping[str, int | str]]] = MappingProxyType(
    {
        "yolo11x": {"max_duration": 300, "label": "xlarge — clips < 5 min"},
        "yolo11l": {"max_duration": 1800, "label": "large — clips 5-30 min"},
        "yolo11m": {"max_duration": 10800, "label": "medium — clips 30 min-3 h"},
        "yolo11s": {"max_duration": 43200, "label": "small — clips 3-12 h"},
        "yolo11n": {"max_duration": 99999, "label": "nano — clips > 12 h"},
    }
)

# Inferencia YOLO
YOLO_CONFIDENCE: float = 0.5
YOLO_NMS_IOU: float = 0.4


# Tracking

TRACKER_MAX_DISTANCE: float = 100.0
TRACKER_MAX_LOST: int = 60
TRACKER_HISTORY_MAXLEN: int = 50


# Flujo óptico

OPTICAL_FLOW_GRID_STEP: int = 40
OPTICAL_FLOW_BORDER_MARGIN: int = 20
OPTICAL_FLOW_WIN_SIZE: tuple[int, int] = (21, 21)
OPTICAL_FLOW_MAX_LEVEL: int = 3
OPTICAL_FLOW_RUNNING_MEAN: int = 30
OPTICAL_FLOW_MIN_TRACKING_RATIO: float = 0.35


# Estimación de velocidad

PIXELS_PER_METER: float = 12.0

# Zonas de perspectiva expresadas como fracción de la altura del frame.
PERSPECTIVE_ZONES: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {
        "near": {"threshold": 0.66, "factor": 1.8},
        "mid": {"threshold": 0.33, "factor": 1.0},
        "far": {"threshold": 0.0, "factor": 0.6},
    }
)

# La banda lineal evita saltos cuando un vehículo cruza límites de perspectiva.
PERSPECTIVE_BLEND_BAND: float = 0.05

# Fusión auxiliar: resultado = peso físico × física + peso MLP × MLP.
SPEED_PHYSICS_WEIGHT: float = 0.70
SPEED_MLP_WEIGHT: float = 0.30
SPEED_MLP_VALID_RANGE: tuple[float, float] = (5.0, 100.0)
SPEED_RECOVERY_SKIP_GAP: int = 1
SPEED_ROBUST_TRIM_RATIO: float = 0.15
SPEED_ROBUST_OUTLIER_SIGMA: float = 3.5

SPEED_MIN_TRACK_LENGTH: int = 8

# Ventana aproximada de un segundo a 30 FPS.
SPEED_ESTIMATION_WINDOW: int = 30

# Los desplazamientos menores al piso de ruido se anulan para evitar velocidades
# producidas por jitter del tracker.
SPEED_DISPLACEMENT_NOISE_FLOOR: float = 2.0

# Límites plausibles por tipo de vehículo, en km/h.
SPEED_LIMITS_PER_TYPE: Final[Mapping[str, tuple[float, float]]] = MappingProxyType(
    {
        "car": (2.0, 120.0),
        "truck": (2.0, 90.0),
        "bus": (2.0, 80.0),
        "motorcycle": (2.0, 130.0),
        "bicycle": (2.0, 40.0),
    }
)


# Movimiento casi nulo: condición más amplia que estacionario.

NEAR_ZERO_TOTAL_DISP_MAX: float = 12.0
NEAR_ZERO_MAX_SEGMENT_MAX: float = 6.0
NEAR_ZERO_STD_MAX: float = 4.0
NEAR_ZERO_AVG_FRAME_MAX: float = 1.2
NEAR_ZERO_MAX_FRAME_MAX: float = 3.5


# Estacionario exige la conjunción de todos los límites siguientes.

STATIONARY_TOTAL_DISP_MAX: float = 5.0
STATIONARY_MAX_SEGMENT_MAX: float = 3.0
STATIONARY_STD_MAX: float = 2.5
STATIONARY_AVG_FRAME_MAX: float = 0.3
STATIONARY_MAX_FRAME_MAX: float = 1.5
STATIONARY_ENTRY_FRAMES: int = 2
STATIONARY_EXIT_FRAMES: int = 3
STATIONARY_EXIT_SPEED_MIN: float = 6.0


# Formato estricto: bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4
VIDEO_FILENAME_PATTERN: str = (
    r"^bridge_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_to_\d{2}-\d{2}-\d{2}\.mp4$"
)
