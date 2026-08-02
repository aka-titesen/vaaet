"""Dataset grouping and split helpers for the academic training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vaaet.settings import RANDOM_SEED

__all__ = [
    "GroupedSplit",
    "RAW_TELEMETRY_COLUMNS",
    "build_group_ids",
    "group_aware_train_test_split",
    "merge_raw_telemetry_csv",
]


@dataclass(frozen=True)
class GroupedSplit:
    """Indices and resolved group ids used in a leakage-aware split."""

    train_idx: np.ndarray
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

            timestamps = pd.to_datetime(df[time_col], errors="coerce")
            fallback = timestamps.dt.floor(fallback_window).astype(str)
            fallback = fallback.where(~timestamps.isna(), row_fallback)
            return groups.mask(groups.eq(""), "window_" + fallback)

    if time_col in df.columns:
        timestamps = pd.to_datetime(df[time_col], errors="coerce")
        fallback = timestamps.dt.floor(fallback_window).astype(str)
        fallback = fallback.where(~timestamps.isna(), row_fallback)
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
    train_idx = np.array([idx for idx, grp in zip(df.index, groups) if grp not in test_group_set])
    test_idx = np.array([idx for idx, grp in zip(df.index, groups) if grp in test_group_set])
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
RAW_TELEMETRY_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "record_time",
    "avg_speed",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "total_vehicles",
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
    missing = set(RAW_TELEMETRY_COLUMNS) - set(telemetry.columns)
    if missing:
        raise ValueError(f"Raw telemetry is missing required columns: {sorted(missing)}")

    frames = [telemetry.copy()]
    if destination.is_file():
        frames.insert(0, pd.read_csv(destination))

    merged = pd.concat(frames, ignore_index=True)
    merged["record_time"] = pd.to_datetime(merged["record_time"], errors="raise")
    merged = (
        merged.drop_duplicates(subset=["clip_id", "record_time"], keep="last")
        .sort_values(["record_time", "clip_id"])
        .reset_index(drop=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    return merged
