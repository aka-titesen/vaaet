# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO detection wrapper for the VAAET production pipeline.

Provides a thin, testable interface around Ultralytics YOLO for vehicle
detection.  The wrapper handles model loading, inference, and post-processing
(NMS filtering, class filtering to vehicles only).

Also includes adaptive model variant selection based on video duration. See
ADR-0002 and ADR-0009 for the decision context.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np

from vaaet.logging import get_logger
from vaaet.settings import YOLO_CONFIDENCE, YOLO_MODEL_VARIANTS, YOLO_NMS_IOU

logger = get_logger(__name__)

__all__ = [
    "Detection",
    "YOLODetector",
    "select_model_variant",
]

# Índices COCO habilitados por el contrato de telemetría vehicular.
_COCO_VEHICLE_IDS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}


@dataclass(frozen=True)
class Detection:
    """Una detección vehicular normalizada para tracking."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    vehicle_type: str
    confidence: float
    centroid: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        cx = (self.bbox[0] + self.bbox[2]) // 2
        cy = (self.bbox[1] + self.bbox[3]) // 2
        object.__setattr__(self, "centroid", (cx, cy))


class _YOLOModel(Protocol):
    """Superficie mínima del runtime Ultralytics aislada del dominio."""

    def __call__(self, frame: np.ndarray, **options: object) -> Iterable[object]:
        """Ejecuta inferencia sobre un frame BGR."""


class YOLODetector:
    """Adaptador acotado de Ultralytics YOLO para detección vehicular.

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
        self._model: _YOLOModel | None = None

    def load(self) -> None:
        """Descarga si hace falta y carga el modelo YOLO seleccionado."""
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

        # Ultralytics no expone tipos estables; el cast permanece en este borde.
        self._model = cast(_YOLOModel, YOLO(f"{self.model_variant}.pt"))
        logger.info("Modelo YOLO cargado: variant=%s", self.model_variant)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Ejecuta detección sobre un frame BGR.

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
            boxes = cast(Iterable[object], getattr(result, "boxes", ()))
            for box in boxes:
                class_values = cast(Iterable[object], getattr(box, "cls", ()))
                confidence_values = cast(Iterable[object], getattr(box, "conf", ()))
                coordinates = cast(Iterable[Iterable[object]], getattr(box, "xyxy", ()))
                class_id = next(iter(class_values), None)
                confidence = next(iter(confidence_values), None)
                bounding_box = next(iter(coordinates), None)
                if class_id is None or confidence is None or bounding_box is None:
                    continue
                cls_id = int(class_id)
                if cls_id not in _COCO_VEHICLE_IDS:
                    continue
                try:
                    x1, y1, x2, y2 = (int(value) for value in bounding_box)
                except (TypeError, ValueError):
                    continue
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        vehicle_type=_COCO_VEHICLE_IDS[cls_id],
                        confidence=float(confidence),
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
