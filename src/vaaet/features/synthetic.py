"""Synthetic edge-case sequences for the VAAET traffic-state classifier.

The real Belgrano Bridge dataset (~2 000 records, Apr–Jul 2025) contains
only Normal and Reduced traffic. Congested and Accident states never
occurred during that period (min speed 5.4 km/h, max vehicles 28).

This module generates physically plausible synthetic sequences that push
the telemetry into Congested (2) and Accident (3) territory so the
classifier can learn all four classes. Synthetic records are clearly
distinguishable from real data:

* ``id`` starts at :data:`SYNTHETIC_ID_OFFSET` (50 001).
* ``record_time`` falls in the week *before* real data (2025-04-21 … 27).
* Provenance columns explicitly label real vs synthetic support.
* ``clip_id`` identifies each synthetic episode so grouped evaluation can
  keep entire episodes together.

The generator is deterministic for a given seed, ensuring reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaaet.settings import (
    DATA_ORIGIN_COL,
    LABELING_THRESHOLDS,
    RANDOM_SEED,
    SYNTHETIC_SCENARIO_COL,
)

__all__ = [
    "augment_with_synthetic",
    "generate_accident_sequences",
    "generate_congestion_sequences",
    "SYNTHETIC_ID_OFFSET",
]

SYNTHETIC_ID_OFFSET: int = 50_001
"""First ``id`` assigned to synthetic records (avoids collision with real IDs 1–2 114)."""

_SYNTHETIC_TIME_START = pd.Timestamp("2025-04-21 06:00:00")
"""Anchor timestamp for synthetic records (1 week before real data)."""


def _with_provenance(
    df: pd.DataFrame,
    *,
    data_origin: str,
    synthetic_scenario: str,
) -> pd.DataFrame:
    """Attach explicit provenance columns without mutating the input."""
    out = df.copy()
    out[DATA_ORIGIN_COL] = data_origin
    out[SYNTHETIC_SCENARIO_COL] = synthetic_scenario
    return out


# Accident sequences


def generate_accident_sequences(
    n_sequences: int = 5,
    records_per_seq: int = 10,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create synthetic telemetry that triggers the Accident (3) label."""
    rng = np.random.default_rng(seed)
    t = LABELING_THRESHOLDS
    rows: list[dict] = []
    base_id = SYNTHETIC_ID_OFFSET
    ts = _SYNTHETIC_TIME_START

    for seq in range(n_sequences):
        clip_id = f"synthetic_accident_{seq + 1:03d}"
        n_approach = max(2, int(records_per_seq * 0.4))

        for i in range(records_per_seq):
            rec: dict[str, object] = {"clip_id": clip_id}

            if i < n_approach:
                rec["avg_speed"] = round(rng.uniform(15, 30), 2)
                cars = int(rng.integers(3, 10))
                trucks = int(rng.integers(0, 3))
                buses = int(rng.integers(0, 2))
                nzm_pct = 0.0
                sc_pct = 0.0
            elif i == n_approach:
                rec["avg_speed"] = round(rng.uniform(0, t["accident_speed_max"]), 2)
                cars = int(rng.integers(0, 3))
                trucks = int(rng.integers(0, 2))
                buses = 0
                nzm_pct = round(rng.uniform(0.1, 0.4), 2)
                sc_pct = 0.0
            else:
                rec["avg_speed"] = round(rng.uniform(0, 1.5), 2)
                cars = int(rng.integers(0, 2))
                trucks = 0
                buses = 0
                nzm_pct = round(rng.uniform(0.5, 1.0), 2)
                sc_pct = round(rng.uniform(0.2, 0.8), 2)

            motos = int(rng.integers(0, 2))
            bikes = int(rng.integers(0, 1))
            total = cars + trucks + buses + motos + bikes

            rec.update(
                {
                    "id": base_id,
                    "record_time": ts,
                    "count_car": cars,
                    "count_truck": trucks,
                    "count_bus": buses,
                    "count_motorcycle": motos,
                    "count_bicycle": bikes,
                    "total_vehicles": max(total, 1),
                    "near_zero_motion_count": int(max(total, 1) * nzm_pct),
                    "stationary_confirmed_count": int(max(total, 1) * sc_pct),
                    "speed_sample_count": max(total, 1),
                    "speed_measurement_quality": round(rng.uniform(0.7, 1.0), 2),
                }
            )
            rows.append(rec)
            base_id += 1
            ts += pd.Timedelta(minutes=1)

        ts += pd.Timedelta(minutes=rng.integers(5, 30))

    return _with_provenance(
        pd.DataFrame(rows),
        data_origin="synthetic",
        synthetic_scenario="accident",
    )


# Congestion sequences


def generate_congestion_sequences(
    n_sequences: int = 5,
    records_per_seq: int = 10,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create synthetic telemetry that triggers the Congested (2) label."""
    rng = np.random.default_rng(seed + 1)
    t = LABELING_THRESHOLDS
    rows: list[dict] = []

    base_id = SYNTHETIC_ID_OFFSET + 500
    ts = _SYNTHETIC_TIME_START + pd.Timedelta(days=3)

    speed_lo = t["accident_speed_max"] + 0.5
    speed_hi = t["congested_speed_max"] - 0.5
    veh_min = int(t["congested_vehicles_min"]) + 2

    for seq in range(n_sequences):
        clip_id = f"synthetic_congestion_{seq + 1:03d}"
        for _ in range(records_per_seq):
            speed = round(rng.uniform(speed_lo, speed_hi), 2)
            cars = int(rng.integers(5, 15))
            trucks = int(rng.integers(1, 5))
            buses = int(rng.integers(0, 3))
            motos = int(rng.integers(0, 3))
            bikes = int(rng.integers(0, 2))
            total = cars + trucks + buses + motos + bikes
            total = max(total, veh_min)

            rows.append(
                {
                    "id": base_id,
                    "clip_id": clip_id,
                    "record_time": ts,
                    "avg_speed": speed,
                    "count_car": cars,
                    "count_truck": trucks,
                    "count_bus": buses,
                    "count_motorcycle": motos,
                    "count_bicycle": bikes,
                    "total_vehicles": total,
                    "near_zero_motion_count": int(total * round(rng.uniform(0.1, 0.3), 2)),
                    "stationary_confirmed_count": int(total * round(rng.uniform(0.0, 0.15), 2)),
                    "speed_sample_count": total,
                    "speed_measurement_quality": round(rng.uniform(0.7, 1.0), 2),
                }
            )
            base_id += 1
            ts += pd.Timedelta(minutes=1)

        ts += pd.Timedelta(minutes=rng.integers(5, 30))

    return _with_provenance(
        pd.DataFrame(rows),
        data_origin="synthetic",
        synthetic_scenario="congestion",
    )


# Public API — augment a real DataFrame


def augment_with_synthetic(
    df_raw: pd.DataFrame,
    *,
    n_accident_seq: int = 10,
    n_congestion_seq: int = 10,
    records_per_seq: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Append synthetic Accident and Congestion sequences to real telemetry."""
    tagged_real = _with_provenance(
        df_raw,
        data_origin="real",
        synthetic_scenario="observed",
    )
    accidents = generate_accident_sequences(
        n_sequences=n_accident_seq,
        records_per_seq=records_per_seq,
        seed=seed,
    )
    congestion = generate_congestion_sequences(
        n_sequences=n_congestion_seq,
        records_per_seq=records_per_seq,
        seed=seed,
    )

    all_cols = list(
        dict.fromkeys(
            [
                *tagged_real.columns,
                *accidents.columns,
                *congestion.columns,
            ]
        )
    )
    result = pd.concat(
        [
            tagged_real.reindex(columns=all_cols),
            accidents.reindex(columns=all_cols),
            congestion.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )
    return result
