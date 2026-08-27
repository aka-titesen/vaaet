"""Tests for src/dataset.py — leakage-aware train/test splitting."""

from __future__ import annotations

import pandas as pd

from vaaet.data.datasets import (
    CANONICAL_RAW_TELEMETRY_COLUMNS,
    build_group_ids,
    group_aware_train_test_split,
    merge_raw_telemetry_csv,
)


def _raw_row(record_time: str, *, speed: float = 12.0) -> dict[str, object]:
    return {
        "clip_id": "bridge_test",
        "record_time": record_time,
        "avg_speed": speed,
        "count_car": 1,
        "count_truck": 0,
        "count_bus": 0,
        "count_motorcycle": 0,
        "count_bicycle": 0,
        "total_vehicles": 1,
    }


def test_merge_raw_telemetry_csv_is_cumulative_and_deduplicated(tmp_path) -> None:
    destination = tmp_path / "traffic_data_raw.csv"
    first = pd.DataFrame([_raw_row("2025-05-01 08:01:00")])
    second = pd.DataFrame(
        [
            _raw_row("2025-05-01 08:01:00", speed=15.0),
            _raw_row("2025-05-01 08:02:00"),
        ]
    )

    merge_raw_telemetry_csv(first, destination)
    merged = merge_raw_telemetry_csv(second, destination)

    assert len(merged) == 2
    assert merged.loc[0, "avg_speed"] == 15.0
    assert destination.is_file()


def test_merge_deduplicates_equivalent_local_and_utc_timestamps(tmp_path) -> None:
    destination = tmp_path / "traffic_data_raw.csv"
    local = pd.DataFrame([_raw_row("2025-05-01 08:01:00", speed=12.0)])
    utc = pd.DataFrame([_raw_row("2025-05-01 11:01:00Z", speed=18.0)])

    merge_raw_telemetry_csv(local, destination)
    merged = merge_raw_telemetry_csv(utc, destination)

    assert len(merged) == 1
    assert merged.iloc[0]["avg_speed"] == 18.0
    assert merged.iloc[0]["record_time"] == pd.Timestamp("2025-05-01 11:01:00Z")


def test_merge_empty_raw_telemetry_does_not_create_csv(tmp_path) -> None:
    destination = tmp_path / "traffic_data_raw.csv"

    merged = merge_raw_telemetry_csv(pd.DataFrame(), destination)

    assert merged.empty
    assert tuple(merged.columns) == CANONICAL_RAW_TELEMETRY_COLUMNS
    assert not destination.exists()


def test_merge_empty_raw_telemetry_does_not_modify_existing_csv(tmp_path) -> None:
    destination = tmp_path / "traffic_data_raw.csv"
    merge_raw_telemetry_csv(
        pd.DataFrame([_raw_row("2025-05-01 08:01:00")]),
        destination,
    )
    original = destination.read_bytes()

    merged = merge_raw_telemetry_csv(pd.DataFrame(), destination)

    assert len(merged) == 1
    assert destination.read_bytes() == original


def test_merge_non_empty_invalid_raw_telemetry_is_rejected(tmp_path) -> None:
    destination = tmp_path / "traffic_data_raw.csv"

    try:
        merge_raw_telemetry_csv(pd.DataFrame([{"clip_id": "incomplete"}]), destination)
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("Invalid non-empty telemetry must be rejected")


class TestBuildGroupIds:
    def test_prefers_clip_id_when_available(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["clip_a", "clip_a", "clip_b"],
                "record_time": pd.date_range("2025-05-01", periods=3, freq="1min"),
            }
        )
        groups = build_group_ids(df)
        assert list(groups) == ["clip_a", "clip_a", "clip_b"]

    def test_falls_back_to_time_windows(self) -> None:
        df = pd.DataFrame(
            {
                "record_time": pd.to_datetime(
                    [
                        "2025-05-01 08:00:00",
                        "2025-05-01 08:04:00",
                        "2025-05-01 08:20:00",
                    ]
                )
            }
        )
        groups = build_group_ids(df, fallback_window="15min")
        assert groups.iloc[0] == groups.iloc[1]
        assert groups.iloc[0] != groups.iloc[2]


class TestGroupAwareTrainTestSplit:
    def test_keeps_groups_disjoint(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["a", "a", "b", "b", "c", "c"],
                "record_time": pd.date_range("2025-05-01", periods=6, freq="1min"),
                "traffic_state": [0, 0, 1, 1, 2, 2],
            }
        )
        split = group_aware_train_test_split(df, test_size=0.33)
        train_groups = set(split.groups.iloc[split.train_idx])
        test_groups = set(split.groups.iloc[split.test_idx])
        assert train_groups.isdisjoint(test_groups)
        assert len(split.train_idx) > 0
        assert len(split.test_idx) > 0
