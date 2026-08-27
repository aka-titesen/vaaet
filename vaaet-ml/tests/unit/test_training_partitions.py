from __future__ import annotations

import pandas as pd
import pytest

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.settings import FEATURE_COLS
from vaaet.training.holdout import HumanHoldoutConfig, resolve_human_holdout
from vaaet.training.lifecycle import TrainingMode
from vaaet.training.partitions import build_training_partitions


def _frame(*, human: bool, prefix: str, state_offset: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for clip_number in range(10):
        state = (clip_number + state_offset) % 3
        for minute in range(3):
            row: dict[str, object] = {
                "clip_id": f"{prefix}-{clip_number}",
                "record_time": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(days=clip_number, minutes=minute),
                "traffic_state": state,
                "is_human_validated": human,
                "data_origin": "real",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
            row.update({feature: float(state + minute + 1) for feature in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_hitl_holdouts_are_human_only_and_exclude_proxy_groups() -> None:
    human = _frame(human=True, prefix="shared")
    proxy = _frame(human=False, prefix="shared")
    partitions = build_training_partitions(
        proxy, human, TrainingMode.HITL_RETRAINING, random_state=42
    )

    assert partitions.validation["is_human_validated"].all()
    assert partitions.test["is_human_validated"].all()
    holdout_groups = set(partitions.validation["clip_id"]) | set(partitions.test["clip_id"])
    assert not (set(partitions.train["clip_id"]) & holdout_groups)


def test_seed_evaluation_contains_no_synthetic_rows() -> None:
    proxy = _frame(human=False, prefix="seed")
    synthetic = proxy.iloc[:3].copy()
    synthetic["clip_id"] = "synthetic-congested"
    synthetic["data_origin"] = "synthetic"
    proxy = pd.concat([proxy, synthetic], ignore_index=True)

    partitions = build_training_partitions(
        proxy, proxy.head(0), TrainingMode.SEED_BOOTSTRAP, random_state=42
    )
    assert not partitions.validation["data_origin"].eq("synthetic").any()
    assert not partitions.test["data_origin"].eq("synthetic").any()
    assert partitions.train["data_origin"].eq("synthetic").any()


def test_hitl_requires_validated_feedback() -> None:
    proxy = _frame(human=False, prefix="seed")
    with pytest.raises(ValueError, match="human-validated"):
        build_training_partitions(
            proxy, proxy.head(0), TrainingMode.HITL_RETRAINING, random_state=42
        )


def test_frozen_holdout_is_exact_and_excludes_all_reserved_groups(tmp_path) -> None:
    human = _frame(human=True, prefix="human")
    snapshot = resolve_human_holdout(
        human,
        HumanHoldoutConfig(store_root=tmp_path, random_state=42),
    )
    proxy = _frame(human=False, prefix="human")
    partitions = build_training_partitions(
        proxy,
        human,
        TrainingMode.HITL_RETRAINING,
        random_state=42,
        frozen_holdout=snapshot,
    )

    assert partitions.validation.equals(snapshot.validation)
    assert partitions.test.equals(snapshot.test)
    assert not (set(partitions.train["clip_id"]) & set(snapshot.reserved_groups))
