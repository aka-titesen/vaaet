"""Canonical UTC and bridge-local timestamp contract tests."""

from __future__ import annotations

import pandas as pd
import pytest

from vaaet.data.timestamps import (
    count_naive_timestamps,
    normalize_timestamp,
    normalize_timestamp_series,
    traffic_local_hour,
)


def test_naive_legacy_timestamp_is_interpreted_as_buenos_aires() -> None:
    result = normalize_timestamp("2025-04-28 10:00:00")
    assert result == pd.Timestamp("2025-04-28 13:00:00Z")


def test_aware_timestamp_preserves_the_same_instant() -> None:
    result = normalize_timestamp("2025-04-28 10:00:00-03:00")
    assert result == pd.Timestamp("2025-04-28 13:00:00Z")


def test_mixed_series_is_utc_and_equivalent_offsets_match() -> None:
    result = normalize_timestamp_series(
        pd.Series(["2025-04-28 10:00:00", "2025-04-28 13:00:00Z"])
    )
    assert str(result.dtype) == "datetime64[ns, UTC]"
    assert result.iloc[0] == result.iloc[1]


def test_local_hour_is_derived_from_canonical_utc() -> None:
    hours = traffic_local_hour(pd.Series([pd.Timestamp("2025-04-28 13:00:00Z")]))
    assert hours.iloc[0] == 10


def test_naive_timestamp_count_is_auditable() -> None:
    values = pd.Series(
        ["2025-04-28 10:00:00", "2025-04-28 13:00:00Z", pd.NaT]
    )
    assert count_naive_timestamps(values) == 1


@pytest.mark.parametrize("value", [None, pd.NaT, "not-a-timestamp"])
def test_invalid_timestamp_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="valid timestamp"):
        normalize_timestamp(value)
