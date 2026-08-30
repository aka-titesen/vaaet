# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Clasificación jerárquica de tránsito compartida entre entrenamiento y serving."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from vaaet.calibration import apply_temperature_scaling
from vaaet.features.engineering import engineer_features
from vaaet.features.labeling import assign_instant_state
from vaaet.inference.policy import (
    CLASSIFICATION_RESULT_COLUMNS,
    apply_conservative_accident_gate,
    apply_stable_state_policy,
    empty_classification_result,
)
from vaaet.inference.protocols import FeatureScaler, TrafficStateModel
from vaaet.lifecycle import ModelInputPolicy, apply_model_input_policy
from vaaet.settings import (
    DEFAULT_MIN_PROBABILITY_MARGIN,
    FEATURE_COLS,
    MODEL_STATE_LABELS,
    MODEL_VERSION,
    RECOVERY_PERSISTENCE_MINUTES,
    STATE_LABELS,
    WORSENING_PERSISTENCE_MINUTES,
)

__all__ = [
    "CLASSIFICATION_RESULT_COLUMNS",
    "apply_conservative_accident_gate",
    "apply_stable_state_policy",
    "classify_raw_telemetry",
    "classify_telemetry_dataframe",
]


def _empty_classification_result(frame: pd.DataFrame) -> pd.DataFrame:
    """Conserva el helper privado para consumidores internos de VAAET 4.x."""

    return empty_classification_result(frame)


def _ensure_feature_compatibility(scaler: FeatureScaler, feature_cols: list[str]) -> None:
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and int(expected) != len(feature_cols):
        raise ValueError(
            "Scaler feature count does not match FEATURE_COLS. Re-run training with bundle v2."
        )


def _validate_probabilities(probabilities: object, rows: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (rows, len(MODEL_STATE_LABELS)):
        raise ValueError(
            "The bundle model must return exactly three probabilities in the order "
            "Normal, Reduced, Congested."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Model probabilities contain invalid values.")
    return values


def classify_telemetry_dataframe(
    df_features: pd.DataFrame,
    model: TrafficStateModel,
    scaler: FeatureScaler,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
    decision_policy: Mapping[str, object] | None = None,
    input_policy: ModelInputPolicy | str = ModelInputPolicy.CANONICAL_V2,
) -> pd.DataFrame:
    """Ejecuta el MLP de tres clases y la cadena de decisión de producción."""

    del label_mapping  # Las etiquetas públicas se gobiernan centralmente.
    if df_features.empty:
        return _empty_classification_result(df_features)
    active_features = feature_cols or FEATURE_COLS
    _ensure_feature_compatibility(scaler, active_features)
    if active_features != FEATURE_COLS:
        raise ValueError("Custom feature order is not supported by bundle v2.")
    model_matrix = apply_model_input_policy(df_features, input_policy)
    probabilities = _validate_probabilities(
        model.predict(scaler.transform(model_matrix.to_numpy()), verbose=0), len(df_features)
    )
    policy = dict(decision_policy or {})
    calibrated_probabilities = apply_temperature_scaling(
        probabilities, float(policy.get("temperature", 1.0))
    )
    result = apply_stable_state_policy(
        df_features,
        calibrated_probabilities,
        class_thresholds=policy.get("class_thresholds"),
        minimum_margin=float(
            policy.get("minimum_probability_margin", DEFAULT_MIN_PROBABILITY_MARGIN)
        ),
        worsening_persistence=int(
            policy.get("worsening_persistence_minutes", WORSENING_PERSISTENCE_MINUTES)
        ),
        recovery_persistence=int(
            policy.get("recovery_persistence_minutes", RECOVERY_PERSISTENCE_MINUTES)
        ),
    )
    result["model_version"] = model_version
    result = apply_conservative_accident_gate(result)
    result["traffic_state"] = result["traffic_state"].astype(int)
    result["state_label"] = result["traffic_state"].map(STATE_LABELS)
    return result


def classify_raw_telemetry(
    df_telemetry: pd.DataFrame,
    model: TrafficStateModel,
    scaler: FeatureScaler,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
    inference_mode: str = "stable",
    decision_policy: Mapping[str, object] | None = None,
    input_policy: ModelInputPolicy | str = ModelInputPolicy.CANONICAL_V2,
) -> pd.DataFrame:
    """Genera features de minutos completos y los clasifica de manera segura."""

    if df_telemetry.empty:
        return _empty_classification_result(df_telemetry)
    if inference_mode == "sprint":
        result = df_telemetry.copy()
        result["traffic_state"] = assign_instant_state(result).astype(int)
        result["state_label"] = result["traffic_state"].map(STATE_LABELS)
        result["confidence"] = 0.0
        result["model_version"] = "sprint_heuristic_non_production"
        result["accident_rule_triggered"] = False
        return result
    features = engineer_features(df_telemetry)
    if features.empty:
        return _empty_classification_result(features)
    return classify_telemetry_dataframe(
        features,
        model,
        scaler,
        label_mapping=label_mapping,
        feature_cols=feature_cols,
        model_version=model_version,
        decision_policy=decision_policy,
        input_policy=input_policy,
    )
