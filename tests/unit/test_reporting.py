"""Tests for src/reporting.py — balance and support summaries."""

from __future__ import annotations

import pandas as pd

from src.reporting import (
    build_class_support_notes,
    summarize_data_origin,
    summarize_resampled_balance,
    summarize_state_balance,
)


class TestSummarizeDataOrigin:
    def test_origin_summary_counts_records(self) -> None:
        df = pd.DataFrame(
            {
                "data_origin": ["real", "real", "synthetic", "synthetic"],
                "synthetic_scenario": [
                    "observed",
                    "observed",
                    "accident",
                    "congestion",
                ],
            }
        )
        summary = summarize_data_origin(df)
        assert int(summary["records"].sum()) == len(df)
        assert set(summary["data_origin"]) == {"real", "synthetic"}


class TestSummarizeStateBalance:
    def test_state_balance_splits_real_and_synthetic(self) -> None:
        df = pd.DataFrame(
            {
                "traffic_state": [0, 0, 2, 3, 3],
                "data_origin": ["real", "real", "synthetic", "synthetic", "synthetic"],
            }
        )
        summary = summarize_state_balance(df)
        accident = summary[summary["traffic_state"] == 3].iloc[0]
        assert accident["real"] == 0
        assert accident["synthetic"] == 2
        assert accident["total"] == 2


class TestSummarizeResampledBalance:
    def test_resampled_delta_is_reported(self) -> None:
        summary = summarize_resampled_balance([0, 0, 1], [0, 0, 1, 1, 1])
        reduced = summary[summary["traffic_state"] == 1].iloc[0]
        assert reduced["before"] == 1
        assert reduced["after"] == 3
        assert reduced["delta"] == 2


class TestBuildClassSupportNotes:
    def test_accident_note_mentions_synthetic_support(self) -> None:
        df = pd.DataFrame(
            {
                "traffic_state": [0, 0, 3, 3],
                "data_origin": ["real", "real", "synthetic", "synthetic"],
            }
        )
        notes = build_class_support_notes(df)
        assert any("Accident" in note and "synthetic" in note for note in notes)

    def test_real_only_data_returns_generic_note(self) -> None:
        df = pd.DataFrame(
            {
                "traffic_state": [0, 0, 1, 1],
                "data_origin": ["real", "real", "real", "real"],
            }
        )
        notes = build_class_support_notes(df)
        assert len(notes) == 1
        assert "real support" in notes[0]
