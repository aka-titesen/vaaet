"""Tests for src/dataset.py — leakage-aware train/test splitting."""

from __future__ import annotations

import pandas as pd

from src.dataset import build_group_ids, group_aware_train_test_split


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
