# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolución determinista de particiones humanas congeladas."""

from __future__ import annotations

import random
from collections.abc import Mapping

import pandas as pd
from vaaet.settings import FEATURE_COLS, MODEL_STATE_LABELS

from vaaet_ml.data.artifact_serialization import sha256_bytes
from vaaet_ml.data.datasets import build_group_ids
from vaaet_ml.training.holdout_contract import (
    IDENTITY_COLUMNS,
    HumanHoldoutAction,
    HumanHoldoutConfig,
    HumanHoldoutSnapshot,
    content_fingerprint,
    csv_bytes,
    prepare_records,
    validate_partition_contract,
)
from vaaet_ml.training.holdout_storage import FileSystemHoldoutStore


def resolve_human_holdout(
    validated_feedback: pd.DataFrame,
    config: HumanHoldoutConfig,
) -> HumanHoldoutSnapshot:
    """Crea, reutiliza o versiona un benchmark humano de validación y test."""

    available = prepare_records(validated_feedback)
    source_fingerprint = sha256_bytes(csv_bytes(available))
    source_groups = set(available["group_id"].astype(str))
    store = FileSystemHoldoutStore(config.store_root)
    current = store.load_current()

    if current is None:
        if config.action is HumanHoldoutAction.CREATE_NEW_VERSION:
            raise ValueError(
                "No frozen holdout exists; use REUSE_OR_CREATE for the first snapshot."
            )
        validation, test = _initial_partitions(available, config)
        return store.write_snapshot(
            validation,
            test,
            generation=1,
            previous_snapshot_id=None,
            update_reason="initial frozen human holdout",
            source_fingerprint=source_fingerprint,
            source_groups=source_groups,
            config=config,
        )

    if config.action is HumanHoldoutAction.REUSE_OR_CREATE:
        conflicts = _snapshot_conflicts(current, available)
        if conflicts:
            raise ValueError(
                "Current human validations contradict the frozen holdout. "
                "Create a new version with an update reason. "
                f"Conflicting keys (sample): {conflicts[:3]}"
            )
        return current

    if current.manifest.get("source_fingerprint") == source_fingerprint:
        return current
    seen_groups = set(current.manifest["source_groups"])
    validation, test = _updated_partitions(
        current, available, config, seen_groups=seen_groups
    )
    if content_fingerprint(validation, test) == current.manifest["fingerprint"]:
        return current
    return store.write_snapshot(
        validation,
        test,
        generation=int(current.manifest["generation"]) + 1,
        previous_snapshot_id=str(current.manifest["snapshot_id"]),
        update_reason=str(config.update_reason).strip(),
        source_fingerprint=source_fingerprint,
        source_groups=seen_groups | source_groups,
        config=config,
    )


def require_comparable_holdouts(
    left: Mapping[str, object] | None,
    right: Mapping[str, object] | None,
) -> None:
    """Rechaza comparar modelos entrenados con snapshots humanos diferentes."""

    if not left or not right:
        raise ValueError("Both models must declare a frozen human holdout.")
    if left.get("fingerprint") != right.get("fingerprint"):
        raise ValueError("Models use different human holdout fingerprints.")


def _initial_partitions(
    frame: pd.DataFrame, config: HumanHoldoutConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_creation_support(frame)
    group_states = {
        str(group): set(group_frame["traffic_state"].astype(int))
        for group, group_frame in frame.groupby("group_id", sort=False)
    }
    group_times = frame.groupby("group_id")["record_time"].max().sort_values(ascending=False)
    ordered_groups = [str(group) for group in group_times.index]
    test_groups = _select_groups(
        ordered_groups,
        group_states,
        desired=max(1, int(round(len(ordered_groups) * config.test_size))),
        remaining_per_state=2,
    )
    validation_candidates = [group for group in ordered_groups if group not in test_groups]
    random.Random(config.random_state).shuffle(validation_candidates)
    validation_groups = _select_groups(
        validation_candidates,
        group_states,
        desired=max(1, int(round(len(ordered_groups) * config.validation_size))),
        remaining_per_state=1,
    )
    validation = frame.loc[frame["group_id"].isin(validation_groups)].copy()
    test = frame.loc[frame["group_id"].isin(test_groups)].copy()
    validate_partition_contract(validation, test)
    return validation, test


def _ensure_creation_support(frame: pd.DataFrame) -> None:
    groups = build_group_ids(frame)
    insufficient = {
        MODEL_STATE_LABELS[state]: int(groups.loc[frame["traffic_state"].eq(state)].nunique())
        for state in MODEL_STATE_LABELS
        if groups.loc[frame["traffic_state"].eq(state)].nunique() < 3
    }
    if insufficient:
        raise ValueError(
            "A frozen holdout requires at least three independent groups per stable state; "
            f"insufficient support: {insufficient}"
        )


def _select_groups(
    candidates: list[str],
    group_states: Mapping[str, set[int]],
    *,
    desired: int,
    remaining_per_state: int,
) -> set[str]:
    selected: set[str] = set()

    def can_reserve(group: str) -> bool:
        remaining = set(candidates) - selected - {group}
        return all(
            sum(state in group_states[item] for item in remaining) >= remaining_per_state
            for state in MODEL_STATE_LABELS
        )

    for state in MODEL_STATE_LABELS:
        if any(state in group_states[group] for group in selected):
            continue
        match = next(
            (
                group
                for group in candidates
                if group not in selected
                and state in group_states[group]
                and can_reserve(group)
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Cannot reserve a leakage-safe holdout group for {MODEL_STATE_LABELS[state]}."
            )
        selected.add(match)
    for group in candidates:
        if len(selected) >= desired:
            break
        if group not in selected and can_reserve(group):
            selected.add(group)
    return selected


def _snapshot_conflicts(
    current: HumanHoldoutSnapshot, available: pd.DataFrame
) -> list[tuple[object, ...]]:
    frozen = _keyed(pd.concat([current.validation, current.test], ignore_index=True))
    candidate = _keyed(available)
    overlap = frozen.index.intersection(candidate.index)
    if overlap.empty:
        return []
    comparison_columns = ["traffic_state", *FEATURE_COLS]
    left = frozen.loc[overlap, comparison_columns].reset_index(drop=True)
    right = candidate.loc[overlap, comparison_columns].reset_index(drop=True)
    numeric_equal = left[FEATURE_COLS].astype(float).eq(right[FEATURE_COLS].astype(float))
    equal = left["traffic_state"].eq(right["traffic_state"]) & numeric_equal.all(axis=1)
    return [tuple(key) if isinstance(key, tuple) else (key,) for key in overlap[~equal]]


def _updated_partitions(
    current: HumanHoldoutSnapshot,
    available: pd.DataFrame,
    config: HumanHoldoutConfig,
    *,
    seen_groups: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = {
        **{group: "validation" for group in current.validation["group_id"].astype(str)},
        **{group: "test" for group in current.test["group_id"].astype(str)},
    }
    updated = {
        partition: _replace_known_groups(current, available, assignments, partition)
        for partition in ("validation", "test")
    }
    new_rows = available.loc[~available["group_id"].isin(seen_groups)].copy()
    if new_rows["group_id"].nunique() >= 3:
        _append_new_groups(updated, new_rows, config)
    return prepare_records(updated["validation"]), prepare_records(updated["test"])


def _replace_known_groups(
    current: HumanHoldoutSnapshot,
    available: pd.DataFrame,
    assignments: Mapping[str, str],
    partition: str,
) -> pd.DataFrame:
    current_frame = getattr(current, partition)
    groups = {group for group, assigned in assignments.items() if assigned == partition}
    replacements = available.loc[available["group_id"].isin(groups)].copy()
    return pd.concat([current_frame, replacements], ignore_index=True).drop_duplicates(
        list(IDENTITY_COLUMNS), keep="last"
    )


def _append_new_groups(
    updated: dict[str, pd.DataFrame], new_rows: pd.DataFrame, config: HumanHoldoutConfig
) -> None:
    ordered_groups = [
        str(group)
        for group in new_rows.groupby("group_id")["record_time"].max().sort_values(ascending=False).index
    ]
    test_count = min(
        len(ordered_groups) - 2,
        max(1, int(round(len(ordered_groups) * config.test_size))),
    )
    test_groups = set(ordered_groups[:test_count])
    remaining = [group for group in ordered_groups if group not in test_groups]
    random.Random(config.random_state).shuffle(remaining)
    validation_count = min(
        len(remaining) - 1,
        max(1, int(round(len(ordered_groups) * config.validation_size))),
    )
    validation_groups = set(remaining[:validation_count])
    updated["validation"] = pd.concat(
        [updated["validation"], new_rows.loc[new_rows["group_id"].isin(validation_groups)]],
        ignore_index=True,
    )
    updated["test"] = pd.concat(
        [updated["test"], new_rows.loc[new_rows["group_id"].isin(test_groups)]],
        ignore_index=True,
    )


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(list(IDENTITY_COLUMNS), drop=False).sort_index()


__all__ = ["require_comparable_holdouts", "resolve_human_holdout"]
