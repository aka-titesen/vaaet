"""Tests for src/reporting.py — balance and support summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaaet_ml.evaluation.reporting import (
    build_class_support_notes,
    build_classification_support_table,
    expected_confusion_cost,
    select_validation_decision_policy,
    summarize_data_origin,
    summarize_resampled_balance,
    summarize_state_balance,
)


def test_classification_support_includes_intervals() -> None:
    table = build_classification_support_table([0, 0, 1, 2], [0, 1, 1, 2])
    assert table["support"].tolist() == [2, 1, 1]
    assert all(len(interval) == 2 for interval in table["recall_ci_95"])


def test_extreme_confusion_costs_more_than_adjacent() -> None:
    assert expected_confusion_cost([0], [2]) > expected_confusion_cost([0], [1])


def test_decision_policy_is_selected_from_validation_probabilities() -> None:
    frame = pd.DataFrame({"clip_id": ["a"] * 4})
    probabilities = np.array(
        [[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.1, 0.85], [0.05, 0.1, 0.85]]
    )
    policy = select_validation_decision_policy(
        frame, [0, 1, 2, 2], probabilities, temperature=1.2
    )
    assert set(policy["class_thresholds"]) == {"0", "1", "2"}
    assert policy["temperature"] == 1.2


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
