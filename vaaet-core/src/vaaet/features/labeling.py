# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Reglas de autoetiquetado para el clasificador de estados de tránsito.

Las etiquetas son supervisión proxy hasta contar con validación HITL. Las
reglas se evalúan por severidad: Accident, Congested, Reduced y Normal.
"""

from __future__ import annotations

import pandas as pd

from vaaet.settings import (
    LABELING_THRESHOLDS,
    NEAR_ZERO_RATIO_MIN,
    SPEED_MEASUREMENT_QUALITY_MIN,
    STATE_LABELS,
    STATIONARY_CONFIRMED_RATIO_MIN,
)

__all__ = [
    "assign_traffic_state",
    "assign_stable_traffic_state",
    "assign_instant_state",
    "build_accident_signal_frame",
    "build_accident_mask",
    "STATE_LABELS",
]


def _optional_ratio(df: pd.DataFrame, column: str, minimum: float) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return values >= minimum


def build_accident_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las señales conservadoras del detector de posibles incidentes."""
    t = LABELING_THRESHOLDS

    low_speed = df["avg_speed"] < float(t["accident_speed_max"])
    braking = df["delta_speed"] < float(t["accident_delta_min"])
    group_key = df["clip_id"] if "clip_id" in df.columns else pd.Series("all", index=df.index)
    cumulative_braking = df["delta_speed"].groupby(group_key, sort=False).transform(
        lambda values: values.rolling(window=int(t["rolling_window"]), min_periods=1).sum()
    )
    had_recent_braking = braking.groupby(group_key, sort=False).transform(
        lambda values: values.rolling(window=int(t["rolling_window"]), min_periods=1)
        .max()
        .astype(bool)
    )
    had_cumulative_braking = cumulative_braking < float(t["accident_cumulative_delta_min"])
    consecutive_low = low_speed.groupby(group_key, sort=False).transform(
        lambda values: values.rolling(
            window=int(t["accident_persistence"]),
            min_periods=int(t["accident_persistence"]),
        ).sum()
    ) >= int(t["accident_persistence"])

    quality_ok = pd.Series(True, index=df.index, dtype=bool)
    if "speed_measurement_quality" in df.columns:
        quality_ok = (
            pd.to_numeric(df["speed_measurement_quality"], errors="coerce")
            .fillna(0.0)
            .ge(SPEED_MEASUREMENT_QUALITY_MIN)
        )

    near_zero_motion = _optional_ratio(df, "near_zero_motion_ratio", NEAR_ZERO_RATIO_MIN)
    stationary_confirmed = _optional_ratio(
        df,
        "stationary_confirmed_ratio",
        STATIONARY_CONFIRMED_RATIO_MIN,
    )
    has_motion_evidence = (
        "near_zero_motion_ratio" in df.columns or "stationary_confirmed_ratio" in df.columns
    )
    motion_evidence = (
        near_zero_motion | stationary_confirmed
        if has_motion_evidence
        else pd.Series(True, index=df.index, dtype=bool)
    )

    evidence_score = (
        low_speed.astype(float) * 0.30
        + had_recent_braking.astype(float) * 0.20
        + had_cumulative_braking.astype(float) * 0.15
        + consecutive_low.astype(float) * 0.20
        + quality_ok.astype(float) * 0.05
        + near_zero_motion.astype(float) * 0.05
        + stationary_confirmed.astype(float) * 0.05
    ).clip(lower=0.0, upper=1.0)

    return pd.DataFrame(
        {
            "accident_low_speed": low_speed,
            "accident_recent_braking": had_recent_braking,
            "accident_cumulative_braking": had_cumulative_braking,
            "accident_persistent_low_speed": consecutive_low,
            "accident_quality_ok": quality_ok,
            "accident_near_zero_motion": near_zero_motion,
            "accident_stationary_confirmed": stationary_confirmed,
            "accident_motion_evidence": motion_evidence,
            "accident_evidence_score": evidence_score,
        },
        index=df.index,
    )


def build_accident_mask(df: pd.DataFrame) -> pd.Series:
    """Combina las señales conservadoras usadas por etiquetado y gates."""
    signals = build_accident_signal_frame(df)
    return (
        signals["accident_low_speed"]
        & (signals["accident_recent_braking"] | signals["accident_cumulative_braking"])
        & signals["accident_persistent_low_speed"]
        & signals["accident_quality_ok"]
        & signals["accident_motion_evidence"]
    )


def assign_instant_state(df: pd.DataFrame) -> pd.Series:
    """Etiqueta clips cortos mediante métricas instantáneas, sin historial.

    Usa umbrales más amplios porque no dispone de persistencia temporal para
    filtrar ruido: menos de 15 km/h es Congested, entre 15 y 30 es Reduced y
    desde 30 es Normal.
    """
    states = pd.Series(0, index=df.index, dtype=int)
    if df.empty or "avg_speed" not in df.columns:
        return states

    congested_mask = (df["avg_speed"] < 15) & (df["total_vehicles"] > 0)
    states[congested_mask] = 2

    reduced_mask = df["avg_speed"].between(15, 30) & (states == 0)
    states[reduced_mask] = 1

    return states


def assign_traffic_state(df: pd.DataFrame) -> pd.Series:
    """Asigna las cuatro etiquetas proxy legadas para comparaciones históricas.

    El entrenamiento nuevo debe usar :func:`assign_stable_traffic_state`.
    Accident se conserva aquí sólo para reproducir experimentos v1.
    """
    t = LABELING_THRESHOLDS
    states = pd.Series(0, index=df.index, dtype=int)

    accident_mask = build_accident_mask(df)
    states[accident_mask] = 3

    congestion = (df["avg_speed"] < t["congested_speed_max"]) & (
        df["total_vehicles"] >= t["congested_vehicles_min"]
    )
    consecutive_congestion = (
        congestion.rolling(
            window=int(t["congested_persistence"]),
            min_periods=int(t["congested_persistence"]),
        ).sum()
        >= t["congested_persistence"]
    )
    stuck_mask = congestion & consecutive_congestion & (states != 3)
    states[stuck_mask] = 2

    reduced_mask = (
        df["avg_speed"].between(t["reduced_speed_min"], t["reduced_speed_max"])
        & df["total_vehicles"].between(
            t["reduced_vehicles_min"],
            t["reduced_vehicles_max"],
        )
        & (states == 0)
    )
    states[reduced_mask] = 1

    return states


def assign_stable_traffic_state(df: pd.DataFrame) -> pd.Series:
    """Asigna únicamente etiquetas proxy Normal, Reduced y Congested.

    Estas reglas inicializan la supervisión y no representan verdad humana.
    La persistencia se limita a cada clip para no mezclar episodios.
    """
    t = LABELING_THRESHOLDS
    states = pd.Series(0, index=df.index, dtype=int)
    congestion = (df["avg_speed"] < t["congested_speed_max"]) & (
        df["total_vehicles"] >= t["congested_vehicles_min"]
    )
    group_key = df["clip_id"] if "clip_id" in df.columns else pd.Series("all", index=df.index)
    consecutive = congestion.groupby(group_key, sort=False).transform(
        lambda values: values.rolling(
            window=int(t["congested_persistence"]),
            min_periods=int(t["congested_persistence"]),
        ).sum()
    ) >= int(t["congested_persistence"])
    states[congestion & consecutive] = 2

    reduced = (
        df["avg_speed"].between(t["reduced_speed_min"], t["reduced_speed_max"])
        & df["total_vehicles"].between(
            t["reduced_vehicles_min"], t["reduced_vehicles_max"]
        )
        & states.eq(0)
    )
    states[reduced] = 1
    return states
