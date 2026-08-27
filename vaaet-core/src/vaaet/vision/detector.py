"""YOLO detection wrapper for the VAAET production pipeline.

Provides a thin, testable interface around Ultralytics YOLO for vehicle
detection.  The wrapper handles model loading, inference, and post-processing
(NMS filtering, class filtering to vehicles only).

Also includes adaptive model variant selection based on video duration. See
ADR-0002 and ADR-0009 for the decision context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vaaet.settings import YOLO_CONFIDENCE, YOLO_MODEL_VARIANTS, YOLO_NMS_IOU

__all__ = [
    "Detection",
    "YOLODetector",
    "select_model_variant",
]

# COCO class indices for vehicle types
_COCO_VEHICLE_IDS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}


@dataclass(frozen=True)
class Detection:
    """A single vehicle detection."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    vehicle_type: str
    confidence: float
    centroid: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        cx = (self.bbox[0] + self.bbox[2]) // 2
        cy = (self.bbox[1] + self.bbox[3]) // 2
        object.__setattr__(self, "centroid", (cx, cy))


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for vehicle detection.

    Args:
        model_variant: YOLO model variant name (e.g. ``"yolo11m"``).
        confidence_threshold: Minimum confidence to keep a detection.
        nms_threshold: IoU threshold for Non-Max Suppression.
    """

    def __init__(
        self,
        model_variant: str = "yolo11m",
        confidence_threshold: float = YOLO_CONFIDENCE,
        nms_threshold: float = YOLO_NMS_IOU,
    ) -> None:
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._model: Any | None = None

    def load(self) -> None:
        """Download (if needed) and load the YOLO model."""
        from ultralytics import YOLO  # deferred import — heavy dependency

        self._model = YOLO(f"{self.model_variant}.pt")
        print(f"✅ YOLO model loaded: {self.model_variant}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image as a numpy array.

        Returns:
            List of :class:`Detection` objects (vehicles only).
        """
        if self._model is None:
            self.load()

        results = self._model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _COCO_VEHICLE_IDS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        vehicle_type=_COCO_VEHICLE_IDS[cls_id],
                        confidence=float(box.conf[0]),
                    )
                )

        return detections


def select_model_variant(duration_seconds: float) -> str:
    """Choose the optimal YOLO model variant based on video duration.

    Shorter clips use lighter models (faster inference); longer clips use
    heavier models (higher accuracy).  Thresholds are defined in
    ``config.YOLO_MODEL_VARIANTS``.

    Args:
        duration_seconds: Clip duration in seconds.

    Returns:
        Model variant name string (e.g. ``"yolo11m"``).
    """
    for variant, meta in YOLO_MODEL_VARIANTS.items():
        if duration_seconds <= meta["max_duration"]:
            return variant
    # Fallback to the largest variant
    return list(YOLO_MODEL_VARIANTS.keys())[-1]
