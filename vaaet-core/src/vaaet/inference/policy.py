# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Políticas de decisión puras para los estados de tránsito e incidentes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vaaet.features.labeling import build_accident_signal_frame
from vaaet.settings import (
    ACCIDENT_GATE_MIN_EVIDENCE_SCORE,
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_MIN_PROBABILITY_MARGIN,
    INCIDENT_PERSISTENCE_MINUTES,
    INCIDENT_RECOVERY_MINUTES,
    MODEL_STATE_LABELS,
    OPTICAL_FLOW_QUALITY_MIN,
    RECOVERY_PERSISTENCE_MINUTES,
    STATE_LABELS,
    WORSENING_PERSISTENCE_MINUTES,
)

CLASSIFICATION_RESULT_COLUMNS: tuple[str, ...] = (
    "model_traffic_state",
    "model_state_label",
    "model_confidence",
    "probability_margin",
    "decision_abstained",
    "traffic_state",
    "state_label",
    "confidence",
    "model_version",
    "accident_low_speed",
    "accident_recent_braking",
    "accident_cumulative_braking",
    "accident_persistent_low_speed",
    "accident_quality_ok",
    "accident_near_zero_motion",
    "accident_stationary_confirmed",
    "accident_motion_evidence",
    "accident_evidence_score",
    "accident_rule_triggered",
    "accident_alert_started",
    "accident_gate_applied",
    "measurement_reliable",
)

_CLASSIFICATION_RESULT_DTYPES: Mapping[str, str] = {
    "model_traffic_state": "Int64",
    "model_state_label": "string",
    "model_confidence": "float64",
    "probability_margin": "float64",
    "decision_abstained": "bool",
    "traffic_state": "Int64",
    "state_label": "string",
    "confidence": "float64",
    "model_version": "string",
    "accident_low_speed": "bool",
    "accident_recent_braking": "bool",
    "accident_cumulative_braking": "bool",
    "accident_persistent_low_speed": "bool",
    "accident_quality_ok": "bool",
    "accident_near_zero_motion": "bool",
    "accident_stationary_confirmed": "bool",
    "accident_motion_evidence": "bool",
    "accident_evidence_score": "float64",
    "accident_rule_triggered": "bool",
    "accident_alert_started": "bool",
    "accident_gate_applied": "bool",
    "measurement_reliable": "bool",
}


def empty_classification_result(frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame vacío que conserva el contrato de clasificación."""
    result = frame.copy()
    for column in CLASSIFICATION_RESULT_COLUMNS:
        if column not in result:
            result[column] = pd.Series(index=result.index, dtype=_CLASSIFICATION_RESULT_DTYPES[column])
    return result


@dataclass
class _StableStateMachine:
    """Estado mínimo para aplicar histéresis dentro de un único clip."""

    stable: int | None = None
    pending: int | None = None
    pending_count: int = 0

    def reset(self) -> None:
        self.stable = None
        self.pending = None
        self.pending_count = 0

    def advance(
        self,
        raw_code: int,
        confidence: float,
        margin: float,
        *,
        thresholds: Mapping[int, float],
        minimum_margin: float,
        worsening_persistence: int,
        recovery_persistence: int,
    ) -> tuple[int, bool]:
        """Aplica una transición por minuto y devuelve estado y abstención."""
        accepted = confidence >= thresholds[raw_code] and margin >= minimum_margin
        if self.stable is None:
            self.stable = min(raw_code, 1) if accepted else 1
            return self.stable, not accepted or raw_code == 2
        if not accepted:
            self.pending = None
            self.pending_count = 0
            return self.stable, True
        target = _adjacent_target(raw_code, self.stable)
        if target == self.stable:
            self.pending = None
            self.pending_count = 0
            return self.stable, False
        self._register_pending(target)
        required = worsening_persistence if target > self.stable else recovery_persistence
        if self.pending_count >= required:
            self.stable = target
            self.pending = None
            self.pending_count = 0
            return self.stable, False
        return self.stable, True

    def _register_pending(self, target: int) -> None:
        if self.pending == target:
            self.pending_count += 1
            return
        self.pending = target
        self.pending_count = 1


def _adjacent_target(target: int, stable: int) -> int:
    if abs(target - stable) <= 1:
        return target
    return stable + (1 if target > stable else -1)


def _probability_summary(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_codes = probabilities.argmax(axis=1).astype(int)
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), raw_codes]
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return raw_codes, confidence, margins


def _validate_policy_probabilities(probabilities: np.ndarray, rows: int) -> np.ndarray:
    if probabilities.shape != (rows, len(MODEL_STATE_LABELS)):
        raise ValueError(
            "The bundle model must return exactly three probabilities in the order "
            "Normal, Reduced, Congested."
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model probabilities contain invalid values.")
    return probabilities


def _stabilize_codes(
    frame: pd.DataFrame,
    raw_codes: np.ndarray,
    confidence: np.ndarray,
    margins: np.ndarray,
    *,
    thresholds: Mapping[int, float],
    minimum_margin: float,
    worsening_persistence: int,
    recovery_persistence: int,
) -> tuple[np.ndarray, np.ndarray]:
    clips = frame.get("clip_id", pd.Series("all", index=frame.index)).astype(str).to_numpy()
    machine = _StableStateMachine()
    prior_clip: str | None = None
    stable_codes = np.empty(len(frame), dtype=int)
    abstained = np.zeros(len(frame), dtype=bool)
    for position, (clip_id, raw_code, score, margin) in enumerate(
        zip(clips, raw_codes, confidence, margins, strict=False)
    ):
        if clip_id != prior_clip:
            prior_clip = clip_id
            machine.reset()
        stable_codes[position], abstained[position] = machine.advance(
            int(raw_code), float(score), float(margin), thresholds=thresholds,
            minimum_margin=minimum_margin, worsening_persistence=worsening_persistence,
            recovery_persistence=recovery_persistence,
        )
    return stable_codes, abstained


def apply_stable_state_policy(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    class_thresholds: Mapping[int, float] | None = None,
    minimum_margin: float = DEFAULT_MIN_PROBABILITY_MARGIN,
    worsening_persistence: int = WORSENING_PERSISTENCE_MINUTES,
    recovery_persistence: int = RECOVERY_PERSISTENCE_MINUTES,
) -> pd.DataFrame:
    """Aplica confianza, adyacencia, persistencia e histéresis por clip."""
    if df.empty:
        return df.copy()
    validated_probabilities = _validate_policy_probabilities(probabilities, len(df))
    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
    if class_thresholds:
        thresholds.update({int(key): float(value) for key, value in class_thresholds.items()})
    raw_codes, raw_confidence, margins = _probability_summary(validated_probabilities)
    stable_codes, abstained = _stabilize_codes(
        df, raw_codes, raw_confidence, margins, thresholds=thresholds,
        minimum_margin=minimum_margin, worsening_persistence=worsening_persistence,
        recovery_persistence=recovery_persistence,
    )
    result = df.copy()
    result["model_traffic_state"] = raw_codes
    result["model_state_label"] = pd.Series(raw_codes, index=result.index).map(MODEL_STATE_LABELS)
    result["model_confidence"] = np.round(raw_confidence, 4)
    result["probability_margin"] = np.round(margins, 4)
    result["decision_abstained"] = abstained
    result["traffic_state"] = stable_codes
    result["state_label"] = pd.Series(stable_codes, index=result.index).map(STATE_LABELS)
    result["confidence"] = np.round(
        validated_probabilities[np.arange(len(result)), stable_codes], 4
    )
    return result


@dataclass
class _IncidentStateMachine:
    """Memoria local que evita publicar accidentes automáticamente."""

    active: bool = False
    weak_count: int = 0

    def reset(self) -> None:
        self.active = False
        self.weak_count = 0

    def advance(self, *, persistent: bool, strong: bool) -> tuple[bool, bool]:
        if persistent:
            started = not self.active
            self.active = True
            self.weak_count = 0
            return self.active, started
        if self.active:
            self.weak_count = 0 if strong else self.weak_count + 1
            if self.weak_count >= INCIDENT_RECOVERY_MINUTES:
                self.reset()
        return self.active, False


def _incident_strength(frame: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    optical_flow = pd.to_numeric(
        frame.get("optical_flow_tracking_ratio", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    quality_reliable = signals["accident_quality_ok"] & optical_flow.ge(OPTICAL_FLOW_QUALITY_MIN)
    strong = (
        signals["accident_evidence_score"].ge(ACCIDENT_GATE_MIN_EVIDENCE_SCORE)
        & signals["accident_persistent_low_speed"]
        & (signals["accident_recent_braking"] | signals["accident_cumulative_braking"])
        & signals["accident_motion_evidence"]
        & quality_reliable
        & pd.to_numeric(frame["total_vehicles"], errors="coerce").fillna(0).gt(0)
    )
    return strong, quality_reliable


def _persistent_incidents(strong: pd.Series, group_key: pd.Series) -> pd.Series:
    return strong.groupby(group_key, sort=False).transform(
        lambda values: values.rolling(
            window=INCIDENT_PERSISTENCE_MINUTES,
            min_periods=INCIDENT_PERSISTENCE_MINUTES,
        ).sum()
    ).ge(INCIDENT_PERSISTENCE_MINUTES)


def _incident_activity(
    clips: np.ndarray, persistent: np.ndarray, strong: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state = _IncidentStateMachine()
    prior_clip: str | None = None
    active_values = np.zeros(len(clips), dtype=bool)
    started_values = np.zeros(len(clips), dtype=bool)
    for position, (clip_id, is_persistent, is_strong) in enumerate(
        zip(clips, persistent, strong, strict=False)
    ):
        if clip_id != prior_clip:
            prior_clip = clip_id
            state.reset()
        active_values[position], started_values[position] = state.advance(
            persistent=bool(is_persistent), strong=bool(is_strong)
        )
    return active_values, started_values


def apply_conservative_accident_gate(
    df: pd.DataFrame,
    *,
    predicted_state_col: str = "traffic_state",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Emite candidatos persistentes sin publicar ``Accident`` automáticamente."""
    if df.empty:
        return df.copy()
    result = df.copy()
    signals = build_accident_signal_frame(result)
    result = pd.concat([result, signals], axis=1)
    strong, quality_reliable = _incident_strength(result, signals)
    group_key = result.get("clip_id", pd.Series("all", index=result.index))
    persistent = _persistent_incidents(strong, group_key)
    active_values, started_values = _incident_activity(
        group_key.astype(str).to_numpy(), persistent.to_numpy(), strong.to_numpy()
    )
    result["accident_rule_triggered"] = active_values
    result["accident_alert_started"] = started_values
    result["accident_gate_applied"] = False
    result["measurement_reliable"] = quality_reliable
    # Un candidato se mantiene como congestión hasta una validación humana explícita.
    result.loc[active_values, predicted_state_col] = 2
    result.loc[active_values, "state_label"] = STATE_LABELS[2]
    result.loc[active_values, confidence_col] = np.maximum(
        pd.to_numeric(result.loc[active_values, confidence_col], errors="coerce").fillna(0.0),
        result.loc[active_values, "accident_evidence_score"],
    ).round(4)
    if result[predicted_state_col].eq(3).any():
        raise RuntimeError("Hierarchical policy invariant violated: automatic Accident state.")
    return result
