from __future__ import annotations

import pandas as pd
import pytest

from vaaet.data.datasets import grouped_temporal_train_validation_test_split
from vaaet.evaluation.dataset_validation import (
    audit_training_dataset,
    validate_training_partitions,
)


def _labeled_groups(raw_telemetry_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_telemetry_df.copy()
    frame["traffic_state"] = frame["clip_id"].str.extract(r"(\d+)")[0].astype(int).clip(upper=2)
    frame["data_origin"] = "real"
    return frame


def test_v2_dataset_is_audited_as_eligible(raw_telemetry_df: pd.DataFrame) -> None:
    audit = audit_training_dataset(raw_telemetry_df)
    assert audit.production_eligible is True
    assert audit.report["telemetry_v2_coverage"] == 1.0
    assert audit.report["timezone"] == "UTC"
    assert audit.report["traffic_local_timezone"] == "America/Argentina/Buenos_Aires"


def test_legacy_dataset_is_not_silently_approved(raw_telemetry_df: pd.DataFrame) -> None:
    legacy = raw_telemetry_df.drop(columns=["telemetry_schema_version"])
    audit = audit_training_dataset(legacy)
    assert audit.production_eligible is False
    with pytest.raises(ValueError, match="not production-eligible"):
        audit_training_dataset(legacy, require_production_eligible=True)


def test_temporal_split_has_no_group_leakage_or_synthetic_evaluation(
    raw_telemetry_df: pd.DataFrame,
) -> None:
    frame = _labeled_groups(raw_telemetry_df)
    synthetic = frame.loc[frame["clip_id"].eq("clip_0")].copy()
    synthetic["clip_id"] = "synthetic_congestion_1"
    synthetic["data_origin"] = "synthetic"
    combined = pd.concat([frame, synthetic], ignore_index=True)
    split = grouped_temporal_train_validation_test_split(combined)
    train = combined.loc[split.train_idx]
    validation = combined.loc[split.validation_idx]
    test = combined.loc[split.test_idx]
    validate_training_partitions(train, validation, test)
    assert train["data_origin"].eq("synthetic").any()
    assert not validation["data_origin"].eq("synthetic").any()
    assert not test["data_origin"].eq("synthetic").any()
    assert pd.to_datetime(test["record_time"]).min() > pd.to_datetime(train["record_time"]).min()


def test_partition_validator_rejects_synthetic_test(raw_telemetry_df: pd.DataFrame) -> None:
    frame = _labeled_groups(raw_telemetry_df)
    train = frame.loc[frame["clip_id"].eq("clip_0")]
    validation = frame.loc[frame["clip_id"].eq("clip_1")]
    test = frame.loc[frame["clip_id"].eq("clip_2")].copy()
    test["data_origin"] = "synthetic"
    with pytest.raises(ValueError, match="forbidden in test"):
        validate_training_partitions(train, validation, test)
