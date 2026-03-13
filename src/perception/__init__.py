# VAAET Perception — YOLO detection, SORT tracking, speed estimation,
# optical-flow camera-motion compensation.

from src.perception.detector import Detection, YOLODetector, select_model_variant
from src.perception.optical_flow import OpticalFlowEstimator
from src.perception.pipeline import MinuteTelemetryAccumulator, process_clip_telemetry
from src.perception.speed import (
    SmoothedSpeedTracker,
    compensate_camera_motion,
    estimate_speed,
    fuse_speed,
    get_perspective_factor,
    is_near_zero_motion,
    is_stationary,
)
from src.perception.tracker import SORTTracker, Track

__all__ = [
    "Detection",
    "MinuteTelemetryAccumulator",
    "OpticalFlowEstimator",
    "SmoothedSpeedTracker",
    "SORTTracker",
    "Track",
    "YOLODetector",
    "compensate_camera_motion",
    "estimate_speed",
    "fuse_speed",
    "get_perspective_factor",
    "is_near_zero_motion",
    "is_stationary",
    "process_clip_telemetry",
    "select_model_variant",
]
