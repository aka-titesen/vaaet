"""Tests for vaaet.settings — single source of truth validation."""

from __future__ import annotations

from types import MappingProxyType


class TestConstants:
    """Verify that critical constants have the expected values and types."""

    def test_random_seed(self) -> None:
        from vaaet.settings import RANDOM_SEED

        assert RANDOM_SEED == 42
        assert isinstance(RANDOM_SEED, int)

    def test_temporal_contract(self) -> None:
        from vaaet.settings import CANONICAL_TIMEZONE, TRAFFIC_LOCAL_TIMEZONE

        assert CANONICAL_TIMEZONE == "UTC"
        assert TRAFFIC_LOCAL_TIMEZONE == "America/Argentina/Buenos_Aires"

    def test_state_labels_immutable(self) -> None:
        from vaaet.settings import STATE_LABELS

        assert isinstance(STATE_LABELS, MappingProxyType)
        assert len(STATE_LABELS) == 4
        assert STATE_LABELS[0] == "Normal"
        assert STATE_LABELS[3] == "Accident"

    def test_n_states(self) -> None:
        from vaaet.settings import N_STATES, STATE_LABELS

        assert N_STATES == len(STATE_LABELS) == 4

    def test_feature_cols_count(self) -> None:
        from vaaet.settings import FEATURE_COLS

        assert len(FEATURE_COLS) == 19
        assert isinstance(FEATURE_COLS, list)
        assert FEATURE_COLS[0] == "avg_speed"
        assert FEATURE_COLS[-1] == "weather_condition"
        assert "speed_measurement_quality" in FEATURE_COLS
        assert "stationary_confirmed_ratio" in FEATURE_COLS

    def test_labeling_thresholds_immutable(self) -> None:
        from vaaet.settings import LABELING_THRESHOLDS

        assert isinstance(LABELING_THRESHOLDS, MappingProxyType)
        assert "accident_speed_max" in LABELING_THRESHOLDS
        assert "accident_cumulative_delta_min" in LABELING_THRESHOLDS
        assert "rolling_window" in LABELING_THRESHOLDS
        assert len(LABELING_THRESHOLDS) == 14

    def test_accident_gate_constants(self) -> None:
        from vaaet.settings import (
            ACCIDENT_GATE_MIN_EVIDENCE_SCORE,
            INCIDENT_PERSISTENCE_MINUTES,
            N_MODEL_STATES,
            SPEED_MEASUREMENT_QUALITY_MIN,
        )

        assert 0.0 < ACCIDENT_GATE_MIN_EVIDENCE_SCORE <= 1.0
        assert 0.0 < SPEED_MEASUREMENT_QUALITY_MIN <= 1.0
        assert INCIDENT_PERSISTENCE_MINUTES >= 2
        assert N_MODEL_STATES == 3

    def test_model_version_format(self) -> None:
        from vaaet.settings import MODEL_VERSION

        assert MODEL_VERSION.startswith("mlp-v")

    def test_speed_range(self) -> None:
        from vaaet.settings import SPEED_RANGE

        assert SPEED_RANGE == (2.0, 120.0)
        assert SPEED_RANGE[0] < SPEED_RANGE[1]

    def test_vehicle_types(self) -> None:
        from vaaet.settings import VEHICLE_TYPES

        assert "car" in VEHICLE_TYPES
        assert "truck" in VEHICLE_TYPES
        assert len(VEHICLE_TYPES) == 5

    def test_db_env_vars(self) -> None:
        from vaaet.settings import DB_ENV_VARS

        assert DB_ENV_VARS == ("VAAET_DB_HOST", "VAAET_DB_PORT", "VAAET_DB_NAME")

    def test_artifact_paths_are_relative(self) -> None:
        import os

        from vaaet.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_DIR

        for path in [MODEL_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR]:
            assert not os.path.isabs(path), f"{path} should be relative"

    def test_provenance_constants(self) -> None:
        from vaaet.settings import (
            DATA_ORIGIN_COL,
            DATA_ORIGINS,
            SYNTHETIC_SCENARIO_COL,
            SYNTHETIC_SCENARIOS,
        )

        assert DATA_ORIGIN_COL == "data_origin"
        assert SYNTHETIC_SCENARIO_COL == "synthetic_scenario"
        assert DATA_ORIGINS == ("real", "synthetic")
        assert SYNTHETIC_SCENARIOS == ("observed", "accident", "congestion")
