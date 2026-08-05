"""Hierarchical, leakage-free traffic-state decisions shared by train and serve."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from vaaet.evaluation.calibration import apply_temperature_scaling
from vaaet.features.engineering import engineer_features
from vaaet.features.labeling import assign_instant_state, build_accident_signal_frame
from vaaet.settings import (
    ACCIDENT_GATE_MIN_EVIDENCE_SCORE,
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_MIN_PROBABILITY_MARGIN,
    FEATURE_COLS,
    INCIDENT_PERSISTENCE_MINUTES,
    INCIDENT_RECOVERY_MINUTES,
    MODEL_STATE_LABELS,
    MODEL_VERSION,
    OPTICAL_FLOW_QUALITY_MIN,
    RECOVERY_PERSISTENCE_MINUTES,
    STATE_LABELS,
    WORSENING_PERSISTENCE_MINUTES,
)

__all__ = [
    "apply_conservative_accident_gate",
    "apply_stable_state_policy",
    "classify_raw_telemetry",
    "classify_telemetry_dataframe",
]


def _ensure_feature_compatibility(scaler: Any, feature_cols: list[str]) -> None:
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and int(expected) != len(feature_cols):
        raise ValueError(
            "Scaler feature count does not match FEATURE_COLS. Re-run training with bundle v2."
        )


def _validate_probabilities(probabilities: Any, rows: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (rows, len(MODEL_STATE_LABELS)):
        raise ValueError(
            "The bundle model must return exactly three probabilities in the order "
            "Normal, Reduced, Congested."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Model probabilities contain invalid values.")
    return values


def apply_stable_state_policy(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    class_thresholds: Mapping[int, float] | None = None,
    minimum_margin: float = DEFAULT_MIN_PROBABILITY_MARGIN,
    worsening_persistence: int = WORSENING_PERSISTENCE_MINUTES,
    recovery_persistence: int = RECOVERY_PERSISTENCE_MINUTES,
) -> pd.DataFrame:
    """Apply confidence, adjacency, persistence, and hysteresis per clip."""
    if df.empty:
        return df.copy()
    proba = _validate_probabilities(probabilities, len(df))
    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
    if class_thresholds:
        thresholds.update({int(key): float(value) for key, value in class_thresholds.items()})

    out = df.copy()
    raw_codes = proba.argmax(axis=1).astype(int)
    sorted_proba = np.sort(proba, axis=1)
    raw_confidence = proba[np.arange(len(out)), raw_codes]
    margins = sorted_proba[:, -1] - sorted_proba[:, -2]
    stable_codes = np.empty(len(out), dtype=int)
    abstained = np.zeros(len(out), dtype=bool)

    clips = out.get("clip_id", pd.Series("all", index=out.index)).astype(str).to_numpy()
    prior_clip: str | None = None
    stable: int | None = None
    pending: int | None = None
    pending_count = 0
    for position, (clip_id, raw_code, confidence, margin) in enumerate(
        zip(clips, raw_codes, raw_confidence, margins)
    ):
        if clip_id != prior_clip:
            prior_clip = clip_id
            stable = None
            pending = None
            pending_count = 0

        accepted = confidence >= thresholds[raw_code] and margin >= minimum_margin
        if stable is None:
            # A first isolated Congested estimate starts safely at Reduced.
            stable = min(int(raw_code), 1) if accepted else 1
            abstained[position] = not accepted or raw_code == 2
            stable_codes[position] = stable
            continue
        if not accepted:
            abstained[position] = True
            stable_codes[position] = stable
            pending = None
            pending_count = 0
            continue

        target = int(raw_code)
        if abs(target - stable) > 1:
            target = stable + (1 if target > stable else -1)
        if target == stable:
            pending = None
            pending_count = 0
        else:
            if pending == target:
                pending_count += 1
            else:
                pending = target
                pending_count = 1
            required = worsening_persistence if target > stable else recovery_persistence
            if pending_count >= required:
                stable = target
                pending = None
                pending_count = 0
            else:
                abstained[position] = True
        stable_codes[position] = stable

    out["model_traffic_state"] = raw_codes
    out["model_state_label"] = pd.Series(raw_codes, index=out.index).map(MODEL_STATE_LABELS)
    out["model_confidence"] = np.round(raw_confidence, 4)
    out["probability_margin"] = np.round(margins, 4)
    out["decision_abstained"] = abstained
    out["traffic_state"] = stable_codes
    out["state_label"] = pd.Series(stable_codes, index=out.index).map(STATE_LABELS)
    out["confidence"] = np.round(proba[np.arange(len(out)), stable_codes], 4)
    return out


def apply_conservative_accident_gate(
    df: pd.DataFrame,
    *,
    predicted_state_col: str = "traffic_state",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Emit a persistent incident candidate without automatic Accident state."""
    if df.empty:
        return df.copy()
    out = df.copy()
    signals = build_accident_signal_frame(out)
    out = pd.concat([out, signals], axis=1)

    optical_flow = pd.to_numeric(
        out.get("optical_flow_tracking_ratio", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    quality_reliable = signals["accident_quality_ok"] & optical_flow.ge(
        OPTICAL_FLOW_QUALITY_MIN
    )
    strong = (
        signals["accident_evidence_score"].ge(ACCIDENT_GATE_MIN_EVIDENCE_SCORE)
        & signals["accident_persistent_low_speed"]
        & (signals["accident_recent_braking"] | signals["accident_cumulative_braking"])
        & signals["accident_motion_evidence"]
        & quality_reliable
        & pd.to_numeric(out["total_vehicles"], errors="coerce").fillna(0).gt(0)
    )
    group_key = out.get("clip_id", pd.Series("all", index=out.index))
    persistent = strong.groupby(group_key, sort=False).transform(
        lambda values: values.rolling(
            window=INCIDENT_PERSISTENCE_MINUTES,
            min_periods=INCIDENT_PERSISTENCE_MINUTES,
        ).sum()
    ).ge(INCIDENT_PERSISTENCE_MINUTES)

    active_values = np.zeros(len(out), dtype=bool)
    started_values = np.zeros(len(out), dtype=bool)
    prior_clip: str | None = None
    active = False
    weak_count = 0
    clips = group_key.astype(str).to_numpy()
    for position, (clip_id, is_persistent, is_strong) in enumerate(
        zip(clips, persistent.to_numpy(), strong.to_numpy())
    ):
        if clip_id != prior_clip:
            prior_clip = clip_id
            active = False
            weak_count = 0
        if is_persistent:
            if not active:
                started_values[position] = True
            active = True
            weak_count = 0
        elif active:
            weak_count = 0 if is_strong else weak_count + 1
            if weak_count >= INCIDENT_RECOVERY_MINUTES:
                active = False
                weak_count = 0
        active_values[position] = active

    out["accident_rule_triggered"] = active_values
    out["accident_alert_started"] = started_values
    out["accident_gate_applied"] = False
    out["measurement_reliable"] = quality_reliable
    # A candidate is operationally Congested until a human confirms Accident.
    out.loc[active_values, predicted_state_col] = 2
    out.loc[active_values, "state_label"] = STATE_LABELS[2]
    out.loc[active_values, confidence_col] = np.maximum(
        pd.to_numeric(out.loc[active_values, confidence_col], errors="coerce").fillna(0.0),
        out.loc[active_values, "accident_evidence_score"],
    ).round(4)
    if out[predicted_state_col].eq(3).any():
        raise RuntimeError("Hierarchical policy invariant violated: automatic Accident state.")
    return out


def classify_telemetry_dataframe(
    df_features: pd.DataFrame,
    model: Any,
    scaler: Any,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
    decision_policy: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the three-class MLP and the exact production decision chain."""
    del label_mapping  # Public labels are governed centrally by STATE_LABELS.
    if df_features.empty:
        return df_features.copy()
    active_features = feature_cols or FEATURE_COLS
    _ensure_feature_compatibility(scaler, active_features)
    missing = df_features[active_features].isna().any()
    if missing.any():
        columns = missing[missing].index.tolist()
        raise ValueError(
            "Cannot classify telemetry with unknown feature values. "
            f"Missing/legacy feature columns: {columns}"
        )
    X = scaler.transform(df_features[active_features].to_numpy())
    policy = dict(decision_policy or {})
    probabilities = _validate_probabilities(model.predict(X, verbose=0), len(df_features))
    probabilities = apply_temperature_scaling(
        probabilities, float(policy.get("temperature", 1.0))
    )
    thresholds = policy.get("class_thresholds")
    out = apply_stable_state_policy(
        df_features,
        probabilities,
        class_thresholds=thresholds,
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
    out["model_version"] = model_version
    out = apply_conservative_accident_gate(out)
    out["traffic_state"] = out["traffic_state"].astype(int)
    out["state_label"] = out["traffic_state"].map(STATE_LABELS)
    return out


def classify_raw_telemetry(
    df_telemetry: pd.DataFrame,
    model: Any,
    scaler: Any,
    *,
    label_mapping: Mapping[int, str] | None = None,
    feature_cols: list[str] | None = None,
    model_version: str = MODEL_VERSION,
    inference_mode: str = "stable",
    decision_policy: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Engineer complete minute windows and classify them safely."""
    if df_telemetry.empty:
        return df_telemetry.copy()
    if inference_mode == "sprint":
        out = df_telemetry.copy()
        out["traffic_state"] = assign_instant_state(out).astype(int)
        out["state_label"] = out["traffic_state"].map(STATE_LABELS)
        out["confidence"] = 0.0
        out["model_version"] = "sprint_heuristic_non_production"
        out["accident_rule_triggered"] = False
        return out

    features = engineer_features(df_telemetry)
    if features.empty:
        return features
    return classify_telemetry_dataframe(
        features,
        model,
        scaler,
        label_mapping=label_mapping,
        feature_cols=feature_cols,
        model_version=model_version,
        decision_policy=decision_policy,
    )
