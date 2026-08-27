# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical timestamp handling shared by every VAAET workflow."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from vaaet.settings import CANONICAL_TIMEZONE, TRAFFIC_LOCAL_TIMEZONE

__all__ = [
    "count_naive_timestamps",
    "normalize_timestamp",
    "normalize_timestamp_series",
    "traffic_local_hour",
    "traffic_local_timestamp_series",
]


def normalize_timestamp(
    value: object,
    *,
    naive_timezone: str = TRAFFIC_LOCAL_TIMEZONE,
    field_name: str = "record_time",
) -> pd.Timestamp:
    """Return a UTC-aware timestamp, interpreting legacy naive values locally."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid timestamp")
    try:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                naive_timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
        return timestamp.tz_convert(CANONICAL_TIMEZONE)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} cannot be normalized from {naive_timezone} to "
            f"{CANONICAL_TIMEZONE}"
        ) from exc


def normalize_timestamp_series(
    values: pd.Series | Iterable[object],
    *,
    naive_timezone: str = TRAFFIC_LOCAL_TIMEZONE,
    field_name: str = "record_time",
) -> pd.Series:
    """Normalize a sequence to the canonical ``datetime64[ns, UTC]`` dtype."""
    source = values if isinstance(values, pd.Series) else pd.Series(values)
    normalized: list[pd.Timestamp] = []
    for position, value in source.items():
        try:
            normalized.append(
                normalize_timestamp(
                    value,
                    naive_timezone=naive_timezone,
                    field_name=field_name,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{field_name} is invalid at index {position!r}") from exc
    return pd.Series(
        normalized,
        index=source.index,
        name=source.name,
        dtype=f"datetime64[ns, {CANONICAL_TIMEZONE}]",
    )


def count_naive_timestamps(values: pd.Series | Iterable[object]) -> int:
    """Count valid timestamp values that do not declare a timezone."""
    source = values if isinstance(values, pd.Series) else pd.Series(values)
    count = 0
    for value in source:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            continue
        if not pd.isna(timestamp) and timestamp.tzinfo is None:
            count += 1
    return count


def traffic_local_timestamp_series(
    values: pd.Series | Iterable[object],
) -> pd.Series:
    """Return canonical instants represented in the bridge's local timezone."""
    return normalize_timestamp_series(values).dt.tz_convert(TRAFFIC_LOCAL_TIMEZONE)


def traffic_local_hour(values: pd.Series | Iterable[object]) -> pd.Series:
    """Return the operational local hour while preserving UTC storage semantics."""
    return traffic_local_timestamp_series(values).dt.hour.astype(int)
