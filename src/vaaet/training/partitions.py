"""Leakage-safe partitions for seed bootstrap and HITL retraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from vaaet.data.datasets import (
    build_group_ids,
    grouped_temporal_train_validation_test_split,
)
from vaaet.data.ingestion import compose_supervised_dataset
from vaaet.evaluation.dataset_validation import validate_training_partitions
from vaaet.training.lifecycle import TrainingMode

if TYPE_CHECKING:
    from vaaet.training.holdout import HumanHoldoutSnapshot


@dataclass(frozen=True)
class TrainingPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_training_partitions(
    proxy_features: pd.DataFrame,
    validated_feedback: pd.DataFrame,
    mode: TrainingMode | str,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int,
    frozen_holdout: HumanHoldoutSnapshot | None = None,
) -> TrainingPartitions:
    """Build proxy evaluation or reuse an exact human-only HITL holdout."""
    active_mode = TrainingMode(mode)
    if active_mode is TrainingMode.SEED_BOOTSTRAP:
        if frozen_holdout is not None:
            raise ValueError("Seed bootstrap cannot use a frozen human holdout.")
        supervised = compose_supervised_dataset(proxy_features, validated_feedback)
        split = grouped_temporal_train_validation_test_split(
            supervised,
            target_col="traffic_state",
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
        )
        partitions = TrainingPartitions(
            supervised.loc[split.train_idx].copy(),
            supervised.loc[split.validation_idx].copy(),
            supervised.loc[split.test_idx].copy(),
        )
    else:
        if validated_feedback.empty:
            raise ValueError("HITL retraining requires human-validated stable labels.")
        if frozen_holdout is None:
            split = grouped_temporal_train_validation_test_split(
                validated_feedback,
                target_col="traffic_state",
                test_size=test_size,
                validation_size=validation_size,
                random_state=random_state,
            )
            human_train = validated_feedback.loc[split.train_idx].copy()
            validation = validated_feedback.loc[split.validation_idx].copy()
            test = validated_feedback.loc[split.test_idx].copy()
            holdout_groups = set(build_group_ids(pd.concat([validation, test])))
        else:
            validation = frozen_holdout.validation.copy()
            test = frozen_holdout.test.copy()
            holdout_groups = set(frozen_holdout.reserved_groups)
            human_train = validated_feedback.loc[
                ~build_group_ids(validated_feedback).isin(holdout_groups)
            ].copy()
        proxy_memory = proxy_features.loc[
            ~build_group_ids(proxy_features).isin(holdout_groups)
        ].copy()
        partitions = TrainingPartitions(
            compose_supervised_dataset(proxy_memory, human_train),
            validation,
            test,
        )
    validate_training_partitions(
        partitions.train, partitions.validation, partitions.test
    )
    return partitions


__all__ = ["TrainingPartitions", "build_training_partitions"]
