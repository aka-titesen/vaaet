# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Dataset grouping and split helpers for the academic training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from vaaet.telemetry import (
    BASE_RAW_TELEMETRY_COLUMNS,
    CANONICAL_RAW_TELEMETRY_COLUMNS,
    TELEMETRY_METADATA_COLUMNS,
    TELEMETRY_QUALITY_COLUMNS,
)
from vaaet.timestamps import normalize_timestamp_series

from vaaet_ml.settings import RANDOM_SEED

__all__ = [
    "GroupedSplit",
    "GroupedTrainValidationTestSplit",
    "BASE_RAW_TELEMETRY_COLUMNS",
    "CANONICAL_RAW_TELEMETRY_COLUMNS",
    "TELEMETRY_METADATA_COLUMNS",
    "TELEMETRY_QUALITY_COLUMNS",
    "build_group_ids",
    "group_aware_train_test_split",
    "grouped_temporal_train_validation_test_split",
    "merge_raw_telemetry_csv",
]


@dataclass(frozen=True)
class GroupedSplit:
    """Indices and resolved group ids used in a leakage-aware split."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    groups: pd.Series


@dataclass(frozen=True)
class GroupedTrainValidationTestSplit:
    """Leakage-safe indices for train, validation, and temporal test."""

    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    groups: pd.Series


def _row_fallback(index: pd.Index) -> pd.Series:
    return pd.Series([f"row_{idx}" for idx in index], index=index, dtype="object")


def build_group_ids(
    df: pd.DataFrame,
    *,
    group_col: str = "clip_id",
    time_col: str = "record_time",
    fallback_window: str = "15min",
) -> pd.Series:
    """Return group ids that prefer clip ids and fall back to time windows."""
    if df.empty:
        return pd.Series(dtype="object")

    row_fallback = _row_fallback(df.index)

    if group_col in df.columns:
        groups = df[group_col].fillna("").astype(str)
        if not groups.eq("").all():
            if time_col not in df.columns:
                return groups.mask(groups.eq(""), row_fallback)

            timestamps = normalize_timestamp_series(df[time_col], field_name=time_col)
            fallback = timestamps.dt.floor(fallback_window).astype(str)
            return groups.mask(groups.eq(""), "window_" + fallback)

    if time_col in df.columns:
        timestamps = normalize_timestamp_series(df[time_col], field_name=time_col)
        fallback = timestamps.dt.floor(fallback_window).astype(str)
        return pd.Series("window_" + fallback, index=df.index, dtype="object")

    return row_fallback


def _fallback_group_split(
    df: pd.DataFrame,
    *,
    groups: pd.Series,
    test_size: float,
) -> GroupedSplit:
    unique_groups = list(pd.Index(groups).drop_duplicates())
    if len(unique_groups) < 2:
        indices = np.arange(len(df))
        if len(df) == 1:
            return GroupedSplit(train_idx=indices, test_idx=np.array([], dtype=int), groups=groups)
        split_at = min(len(df) - 1, max(1, int(round(len(df) * (1.0 - test_size)))))
        return GroupedSplit(
            train_idx=indices[:split_at],
            test_idx=indices[split_at:],
            groups=groups,
        )

    test_group_count = min(
        len(unique_groups) - 1,
        max(1, int(round(len(unique_groups) * test_size))),
    )
    test_group_set = set(unique_groups[-test_group_count:])
    train_idx = np.array(
        [idx for idx, grp in zip(df.index, groups, strict=False) if grp not in test_group_set]
    )
    test_idx = np.array(
        [idx for idx, grp in zip(df.index, groups, strict=False) if grp in test_group_set]
    )
    return GroupedSplit(train_idx=train_idx, test_idx=test_idx, groups=groups)


def group_aware_train_test_split(
    df: pd.DataFrame,
    *,
    target_col: str = "traffic_state",
    group_col: str = "clip_id",
    time_col: str = "record_time",
    fallback_window: str = "15min",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> GroupedSplit:
    """Split train/test while keeping clips or time windows intact."""
    if df.empty:
        empty = np.array([], dtype=int)
        return GroupedSplit(train_idx=empty, test_idx=empty, groups=pd.Series(dtype="object"))

    groups = build_group_ids(
        df,
        group_col=group_col,
        time_col=time_col,
        fallback_window=fallback_window,
    )
    y = df[target_col].to_numpy()
    unique_groups = groups.nunique(dropna=False)
    unique_classes = np.unique(y).size

    if unique_groups < 2:
        return _fallback_group_split(df, groups=groups, test_size=test_size)

    try:
        from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    except ModuleNotFoundError:
        return _fallback_group_split(df, groups=groups, test_size=test_size)

    if unique_classes >= 2:
        n_splits = min(5, unique_groups)
        if n_splits >= 2:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            try:
                candidate_splits = list(splitter.split(df, y, groups))
            except ValueError:
                candidate_splits = []
            candidate_splits = [pair for pair in candidate_splits if len(pair[0]) and len(pair[1])]
            if candidate_splits:
                train_idx, test_idx = min(
                    candidate_splits,
                    key=lambda pair: abs((len(pair[1]) / len(df)) - test_size),
                )
                return GroupedSplit(
                    train_idx=np.asarray(train_idx),
                    test_idx=np.asarray(test_idx),
                    groups=groups,
                )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, y, groups))
    return GroupedSplit(
        train_idx=np.asarray(train_idx),
        test_idx=np.asarray(test_idx),
        groups=groups,
    )


def grouped_temporal_train_validation_test_split(
    df: pd.DataFrame,
    *,
    target_col: str = "traffic_state",
    group_col: str = "clip_id",
    time_col: str = "record_time",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> GroupedTrainValidationTestSplit:
    """Reserve latest real groups for test, then split grouped validation.

    Rows marked ``data_origin=synthetic`` are assigned only to train. Returned
    arrays contain dataframe index labels, which keeps provenance traceable.
    """
    if df.empty:
        empty = np.array([], dtype=int)
        return GroupedTrainValidationTestSplit(empty, empty, empty, pd.Series(dtype="object"))
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must sum to less than 1.")

    groups = build_group_ids(df, group_col=group_col, time_col=time_col)
    origins = df.get("data_origin", pd.Series("real", index=df.index))
    synthetic = origins.eq("synthetic")
    real = df.loc[~synthetic]
    if real.empty:
        raise ValueError("At least one real group is required for validation and test.")

    real_groups = groups.loc[real.index]
    timestamps = normalize_timestamp_series(real[time_col], field_name=time_col)
    group_times = timestamps.groupby(real_groups).max().sort_values()
    if len(group_times) < 3:
        raise ValueError("At least three real clips/groups are required for three-way splitting.")
    test_group_count = min(
        len(group_times) - 2,
        max(1, int(np.ceil(len(group_times) * test_size))),
    )
    test_groups = set(group_times.index[-test_group_count:])
    test_idx = real.index[real_groups.isin(test_groups)].to_numpy()

    remaining = real.loc[~real_groups.isin(test_groups)]
    relative_validation = validation_size / (1.0 - test_size)
    val_split = group_aware_train_test_split(
        remaining,
        target_col=target_col,
        group_col=group_col,
        time_col=time_col,
        test_size=relative_validation,
        random_state=random_state,
    )
    train_real_idx = remaining.iloc[val_split.train_idx].index.to_numpy()
    validation_idx = remaining.iloc[val_split.test_idx].index.to_numpy()
    train_idx = np.concatenate([train_real_idx, df.index[synthetic].to_numpy()])

    row_parts = [set(train_idx), set(validation_idx), set(test_idx)]
    if any(
        row_parts[left] & row_parts[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("Dataset split contains overlapping rows.")
    group_parts = [set(groups.loc[list(rows)]) for rows in row_parts]
    if any(
        group_parts[left] & group_parts[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("Dataset split leaks a clip/group across partitions.")
    return GroupedTrainValidationTestSplit(
        train_idx=np.asarray(train_idx),
        validation_idx=np.asarray(validation_idx),
        test_idx=np.asarray(test_idx),
        groups=groups,
    )


def merge_raw_telemetry_csv(
    telemetry: pd.DataFrame,
    destination: str | Path,
) -> pd.DataFrame:
    """Merge telemetry into the canonical CSV and deduplicate idempotently.

    The acquisition contract uses ``(clip_id, record_time)`` as its natural
    key. Extended quality columns are preserved for future feature engineering.
    """
    destination = Path(destination)
    if telemetry.empty:
        return _existing_telemetry_or_empty(destination)

    _require_raw_telemetry_columns(telemetry, source="Raw telemetry")
    frames = [_with_optional_telemetry_columns(telemetry.copy())]
    if destination.is_file():
        frames.insert(0, _read_existing_telemetry(destination))

    merged = pd.concat(frames, ignore_index=True)
    merged["record_time"] = normalize_timestamp_series(merged["record_time"])
    merged = (
        merged.drop_duplicates(subset=["clip_id", "record_time"], keep="last")
        .sort_values(["record_time", "clip_id"])
        .reset_index(drop=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    return merged


def _existing_telemetry_or_empty(destination: Path) -> pd.DataFrame:
    if not destination.is_file():
        return pd.DataFrame(columns=CANONICAL_RAW_TELEMETRY_COLUMNS)
    return _read_existing_telemetry(destination)


def _read_existing_telemetry(destination: Path) -> pd.DataFrame:
    existing = pd.read_csv(destination, float_precision="round_trip")
    _require_raw_telemetry_columns(existing, source="Existing raw telemetry")
    return _with_optional_telemetry_columns(existing)


def _require_raw_telemetry_columns(telemetry: pd.DataFrame, *, source: str) -> None:
    missing = sorted(set(BASE_RAW_TELEMETRY_COLUMNS) - set(telemetry.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _with_optional_telemetry_columns(telemetry: pd.DataFrame) -> pd.DataFrame:
    for column in (*TELEMETRY_QUALITY_COLUMNS, *TELEMETRY_METADATA_COLUMNS):
        if column not in telemetry:
            telemetry[column] = pd.NA
    return telemetry
