"""Synthetic edge-case sequences for the VAAET traffic-state classifier.

The real Belgrano Bridge dataset (~2 000 records, Apr–Jul 2025) contains
only Normal and Reduced traffic.  Congested and Accident states never
occurred during that period (min speed 5.4 km/h, max vehicles 28).

This module generates physically plausible synthetic sequences that push
the telemetry into Congested (2) and Accident (3) territory so the
classifier can learn all four classes.  Synthetic records are clearly
distinguishable from real data:

* ``id`` starts at :data:`SYNTHETIC_ID_OFFSET` (50 001).
* ``record_time`` falls in the week *before* real data (2025-04-21 … 27).

The generator is deterministic for a given seed, ensuring reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABELING_THRESHOLDS, RANDOM_SEED

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


# Accident sequences


def generate_accident_sequences(
    n_sequences: int = 5,
    records_per_seq: int = 10,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create synthetic telemetry that triggers the Accident (3) label.

    Each sequence models a realistic incident on the bridge:

    1. **Approach** (first ~40 %): moderate speed 15–30 km/h, normal volume.
    2. **Impact** (1 record): sudden braking → speed drops to 0–2 km/h.
    3. **Standstill** (remaining ~50 %): speed 0–1.5 km/h, low/zero volume,
       sustained for ≥ ``accident_persistence`` records.

    Args:
        n_sequences: How many independent accident sequences to generate.
        records_per_seq: Records per sequence (≥ 6 recommended).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with the ``traffic_data`` schema (``id``, ``record_time``,
        ``avg_speed``, 5 vehicle counts, ``total_vehicles``).
    """
    rng = np.random.default_rng(seed)
    t = LABELING_THRESHOLDS
    rows: list[dict] = []
    base_id = SYNTHETIC_ID_OFFSET
    ts = _SYNTHETIC_TIME_START

    for seq in range(n_sequences):
        n_approach = max(2, int(records_per_seq * 0.4))
        n_standstill = records_per_seq - n_approach - 1  # 1 for impact

        for i in range(records_per_seq):
            rec: dict = {}

            if i < n_approach:
                # Approach phase — moderate speed
                rec["avg_speed"] = round(rng.uniform(15, 30), 2)
                cars = int(rng.integers(3, 10))
                trucks = int(rng.integers(0, 3))
                buses = int(rng.integers(0, 2))
            elif i == n_approach:
                # Impact — sudden braking to near-zero
                rec["avg_speed"] = round(rng.uniform(0, t["accident_speed_max"]), 2)
                cars = int(rng.integers(0, 3))
                trucks = int(rng.integers(0, 2))
                buses = 0
            else:
                # Standstill — sustained near-zero
                rec["avg_speed"] = round(rng.uniform(0, 1.5), 2)
                cars = int(rng.integers(0, 2))
                trucks = 0
                buses = 0

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
                }
            )
            rows.append(rec)
            base_id += 1
            ts += pd.Timedelta(minutes=1)

        # Small gap between sequences
        ts += pd.Timedelta(minutes=rng.integers(5, 30))

    return pd.DataFrame(rows)


# Congestion sequences


def generate_congestion_sequences(
    n_sequences: int = 5,
    records_per_seq: int = 10,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create synthetic telemetry that triggers the Congested (2) label.

    Each sequence models a traffic jam scenario:

    * Speed between ``accident_speed_max`` and ``congested_speed_max``
      (2–7 km/h by default), with gradual slow-down.
    * Vehicle count above ``congested_vehicles_min`` (≥ 8 by default),
      simulating vehicle accumulation.
    * Duration ≥ ``congested_persistence`` records to satisfy the
      rolling-window criterion.

    Args:
        n_sequences: Independent congestion episodes.
        records_per_seq: Records per episode.
        seed: RNG seed.

    Returns:
        DataFrame with ``traffic_data`` schema.
    """
    rng = np.random.default_rng(seed + 1)  # offset seed to differ from accidents
    t = LABELING_THRESHOLDS
    rows: list[dict] = []

    # Continue IDs after accidents (generous gap)
    base_id = SYNTHETIC_ID_OFFSET + 500
    ts = _SYNTHETIC_TIME_START + pd.Timedelta(days=3)

    speed_lo = t["accident_speed_max"] + 0.5  # > 2 km/h (not accident)
    speed_hi = t["congested_speed_max"] - 0.5  # < 7 km/h (congested territory)
    veh_min = int(t["congested_vehicles_min"]) + 2  # comfortably above threshold

    for seq in range(n_sequences):
        for i in range(records_per_seq):
            speed = round(rng.uniform(speed_lo, speed_hi), 2)
            cars = int(rng.integers(5, 15))
            trucks = int(rng.integers(1, 5))
            buses = int(rng.integers(0, 3))
            motos = int(rng.integers(0, 3))
            bikes = int(rng.integers(0, 2))
            total = cars + trucks + buses + motos + bikes
            total = max(total, veh_min)  # ensure above threshold

            rows.append(
                {
                    "id": base_id,
                    "record_time": ts,
                    "avg_speed": speed,
                    "count_car": cars,
                    "count_truck": trucks,
                    "count_bus": buses,
                    "count_motorcycle": motos,
                    "count_bicycle": bikes,
                    "total_vehicles": total,
                }
            )
            base_id += 1
            ts += pd.Timedelta(minutes=1)

        ts += pd.Timedelta(minutes=rng.integers(5, 30))

    return pd.DataFrame(rows)


# Public API — augment a real DataFrame


def augment_with_synthetic(
    df_raw: pd.DataFrame,
    *,
    n_accident_seq: int = 10,
    n_congestion_seq: int = 10,
    records_per_seq: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Append synthetic Accident and Congestion sequences to real telemetry.

    The returned DataFrame has the same schema as the input, with synthetic
    rows appended at the end.  All synthetic ``id`` values are ≥
    :data:`SYNTHETIC_ID_OFFSET` so they can be trivially filtered out.

    Args:
        df_raw: Real telemetry from ``traffic_data``.
        n_accident_seq: Number of accident episodes to generate.
        n_congestion_seq: Number of congestion episodes to generate.
        records_per_seq: Records per synthetic episode.
        seed: RNG seed.

    Returns:
        Concatenated DataFrame (real + synthetic), index reset.
    """
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

    # Ensure column alignment (synthetic frames have same cols as real)
    common_cols = [c for c in df_raw.columns if c in accidents.columns]
    result = pd.concat(
        [df_raw, accidents[common_cols], congestion[common_cols]],
        ignore_index=True,
    )
    return result
