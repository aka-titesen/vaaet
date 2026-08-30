# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Exportación portable de feedback humano sin widgets ni PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from vaaet_ml.data.package_codec import create_dataset_package
from vaaet_ml.data.review_domain import HumanValidation


def export_offline_review_package(
    output_path: str | Path,
    *,
    classified: pd.DataFrame,
    validations: pd.DataFrame | Sequence[HumanValidation],
) -> Path:
    """Crea un paquete portable sólo cuando existe al menos una decisión humana."""

    features = classified.copy()
    if "id" not in features:
        features["id"] = range(1, len(features) + 1)
    prediction_ids = (
        features["prediction_id"].tolist()
        if "prediction_id" in features
        else list(range(1, len(features) + 1))
    )
    predictions = pd.DataFrame(
        {
            "id": prediction_ids,
            "telemetry_feature_id": features["id"],
            "model_version": features.get("model_version", "unknown"),
        }
    )
    features = features.drop(columns=["prediction_id"], errors="ignore")
    validation_frame = _validation_frame(validations)
    if validation_frame.empty:
        raise ValueError("Complete at least one human validation before exporting feedback.")
    validation_frame["id"] = _validation_ids(validation_frame)
    validation_frame["validated_state"] = validation_frame.pop("validated_state")
    exported_at = pd.Timestamp.now(tz="UTC")
    validation_frame["reviewed_at"] = [
        exported_at + pd.Timedelta(microseconds=index)
        for index in range(len(validation_frame))
    ]
    return create_dataset_package(
        output_path,
        features=features,
        predictions=predictions,
        validations=validation_frame,
        provenance={"origin": "inference-colab-human-review"},
    )


def _validation_frame(
    validations: pd.DataFrame | Sequence[HumanValidation],
) -> pd.DataFrame:
    if isinstance(validations, pd.DataFrame):
        return validations.copy()
    frame = pd.DataFrame([asdict(decision) for decision in validations])
    return frame.rename(columns={"validation_id": "id"})


def _validation_ids(frame: pd.DataFrame) -> list[str]:
    if "id" not in frame:
        return [str(uuid4()) for _ in range(len(frame))]
    return [str(uuid4() if value is None or pd.isna(value) else value) for value in frame["id"]]


__all__ = ["export_offline_review_package"]
