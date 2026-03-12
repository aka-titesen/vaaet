"""Shared classification helpers for VAAET notebooks and modules."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import (
    ACCIDENT_GATE_LOW_CONFIDENCE_MAX,
    ACCIDENT_GATE_MIN_EVIDENCE_SCORE,
    FEATURE_COLS,
    MODEL_VERSION,
    STATE_LABELS,
)
from src.features import engineer_features
from src.labeling import build_accident_signal_frame

__all__ = [
    "apply_conservative_accident_gate",
    "classify_raw_telemetry",
    "classify_telemetry_dataframe",
]


def _resolve_label_mapping(
    label_mapping: Mapping[int, str] | None,
) -> dict[int, str]:
    mapping = dict(STATE_LABELS)
    if label_mapping:
        mapping.update({int(key): value for key, value in label_mapping.items()})
    return mapping


def _ensure_feature_compatibility(
    scaler: Any,
    feature_cols: list[str],
) -> None:
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and int(expected) != len(feature_cols):
        raise ValueError(
            "Scaler feature count does not match FEATURE_COLS. "
            "Re-run Module 1 so the artifacts are regenerated with the current schema."
        )


def apply_conservative_accident_gate(
    df: pd.DataFrame,
    *,
    predicted_state_col: str = "traffic_state",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Override the model with a conservative accident rule when evidence is strong."""
    if df.empty:
        return df.copy()

    out = df.copy()
    signals = build_accident_signal_frame(out)
    out = pd.concat([out, signals], axis=1)

    allowed_override = (
        out[predicted_state_col].isin([2, 3])
        | pd.to_numeric(out[confidence_col], errors="coerce")
        .fillna(0.0)
        .le(ACCIDENT_GATE_LOW_CONFIDENCE_MAX)
    )
    strong_evidence = out["accident_evidence_score"] >= ACCIDENT_GATE_MIN_EVIDENCE_SCORE
    gate = strong_evidence & allowed_override & (
        out["accident_persistent_low_speed"]
        & (
            out["accident_recent_braking"]
            | out["accident_cumulative_braking"]
        )
        & out["accident_motion_evidence"]
    )

    out["accident_rule_triggered"] = strong_evidence
    out["accident_gate_applied"] = gate

    model_state = out[predicted_state_col].astype(int)
    model_confidence = pd.to_numeric(out[confidence_col], errors="coerce").fillna(0.0)

    out["model_traffic_state"] = model_state
    out["model_state_label"] = model_state.map(STATE_LABELS)
    out["model_confidence"] = model_confidence.round(4)

    out.loc[gate, predicted_state_col] = 3
    out.loc[gate, "state_label"] = STATE_LABELS[3]
    out.loc[gate, confidence_col] = np.maximum(
        model_confidence.loc[gate],
        out.loc[gate, "accident_evidence_score"],
    ).round(4)
    return out


def classify_telemetry_dataframe(
    df_features: pd.DataFrame,
    model: Any,
    scaler: Any,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Classify engineered telemetry and apply the conservative accident gate."""
    if df_features.empty:
        return df_features.copy()

    active_feature_cols = feature_cols or FEATURE_COLS
    _ensure_feature_compatibility(scaler, active_feature_cols)
    labels = _resolve_label_mapping(label_mapping)

    out = df_features.copy()
    X = scaler.transform(out[active_feature_cols].values)
    X = np.nan_to_num(X, nan=0.0)

    proba = model.predict(X, verbose=0)
    pred_codes = proba.argmax(axis=1).astype(int)
    confidences = proba.max(axis=1).astype(float)

    out["traffic_state"] = pred_codes
    out["state_label"] = [labels.get(code, STATE_LABELS.get(code, "Unknown")) for code in pred_codes]
    out["confidence"] = np.round(confidences, 4)
    out["model_version"] = model_version

    out = apply_conservative_accident_gate(out)
    out["traffic_state"] = out["traffic_state"].astype(int)
    out["state_label"] = out["traffic_state"].map(STATE_LABELS)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).round(4)
    return out


def classify_raw_telemetry(
    df_telemetry: pd.DataFrame,
    model: Any,
    scaler: Any,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Engineer features from raw telemetry and classify them safely."""
    if df_telemetry.empty:
        return df_telemetry.copy()

    df_feat = engineer_features(df_telemetry)
    if df_feat.empty and not df_telemetry.empty:
        duplicated = pd.concat(
            [df_telemetry.iloc[[0]], df_telemetry.iloc[[0]]],
            ignore_index=True,
        )
        df_feat = engineer_features(duplicated).head(1).copy()

    if df_feat.empty:
        return df_feat

    return classify_telemetry_dataframe(
        df_feat,
        model,
        scaler,
        label_mapping=label_mapping,
        feature_cols=feature_cols,
        model_version=model_version,
    )
