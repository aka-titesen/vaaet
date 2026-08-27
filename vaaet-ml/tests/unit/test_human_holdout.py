# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.settings import FEATURE_COLS
from vaaet_ml.training.holdout import (
    CURRENT_POINTER_FILE,
    FileSystemHoldoutStore,
    HumanHoldoutAction,
    HumanHoldoutConfig,
    require_comparable_holdouts,
    resolve_human_holdout,
)


def _feedback(*, groups: int = 15, start: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number in range(start, start + groups):
        state = number % 3
        for minute in range(2):
            row: dict[str, object] = {
                "id": str(uuid.uuid4()),
                "clip_id": f"human-{number:03d}",
                "record_time": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(days=number, minutes=minute),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "traffic_state": state,
                "is_human_validated": True,
                "reviewer_id": "reviewer-test",
                "reviewed_at": pd.Timestamp("2026-02-01T00:00:00Z"),
            }
            row.update({feature: float(number + minute + 1) for feature in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def _config(root: Path, **kwargs: object) -> HumanHoldoutConfig:
    return HumanHoldoutConfig(
        store_root=root,
        git_commit="a" * 40,
        vaaet_version="4.4.0",
        random_state=42,
        **kwargs,
    )


def test_creates_and_reuses_exact_snapshot(tmp_path: Path) -> None:
    feedback = _feedback()
    first = resolve_human_holdout(feedback, _config(tmp_path))
    second = resolve_human_holdout(
        pd.concat([feedback, _feedback(groups=3, start=30)], ignore_index=True),
        _config(tmp_path),
    )

    assert first.path == second.path
    assert first.descriptor == second.descriptor
    assert (tmp_path / CURRENT_POINTER_FILE).is_file()
    assert len(list(tmp_path.glob("human-holdout-*.zip"))) == 1
    assert set(first.validation["traffic_state"]) == {0, 1, 2}
    assert set(first.test["traffic_state"]) == {0, 1, 2}


def test_update_creates_new_version_without_overwriting(tmp_path: Path) -> None:
    original = _feedback()
    first = resolve_human_holdout(original, _config(tmp_path))
    expanded = pd.concat([original, _feedback(groups=6, start=30)], ignore_index=True)
    update_config = _config(
        tmp_path,
        action=HumanHoldoutAction.CREATE_NEW_VERSION,
        update_reason="Add independently reviewed July clips",
    )
    second = resolve_human_holdout(expanded, update_config)
    repeated = resolve_human_holdout(expanded, update_config)

    assert first.path.is_file()
    assert second.path.is_file()
    assert first.path != second.path
    assert second.manifest["generation"] == 2
    assert second.manifest["previous_snapshot_id"] == first.manifest["snapshot_id"]
    assert repeated.path == second.path
    assert len(list(tmp_path.glob("human-holdout-*.zip"))) == 2


def test_later_updates_do_not_reconsider_groups_already_left_in_train(
    tmp_path: Path,
) -> None:
    original = _feedback()
    first = resolve_human_holdout(original, _config(tmp_path))
    first_expansion = _feedback(groups=6, start=30)
    second = resolve_human_holdout(
        pd.concat([original, first_expansion], ignore_index=True),
        _config(
            tmp_path,
            action=HumanHoldoutAction.CREATE_NEW_VERSION,
            update_reason="First expansion",
        ),
    )
    expansion_groups = set(first_expansion["clip_id"])
    groups_left_in_train = expansion_groups - set(second.reserved_groups)
    assert groups_left_in_train

    third = resolve_human_holdout(
        pd.concat(
            [original, first_expansion, _feedback(groups=3, start=60)],
            ignore_index=True,
        ),
        _config(
            tmp_path,
            action=HumanHoldoutAction.CREATE_NEW_VERSION,
            update_reason="Second expansion",
        ),
    )

    assert first.path.is_file()
    assert not (groups_left_in_train & set(third.reserved_groups))


def test_partial_correction_preserves_other_frozen_minutes(tmp_path: Path) -> None:
    first = resolve_human_holdout(_feedback(), _config(tmp_path))
    correction = first.validation.iloc[[0]].copy()
    correction.loc[:, FEATURE_COLS[0]] = correction[FEATURE_COLS[0]] + 0.5
    before_rows = len(first.validation)

    second = resolve_human_holdout(
        correction,
        _config(
            tmp_path,
            action=HumanHoldoutAction.CREATE_NEW_VERSION,
            update_reason="Correct one reviewed feature snapshot",
        ),
    )

    assert len(second.validation) == before_rows
    assert second.manifest["generation"] == 2


def test_reuse_detects_changed_effective_label(tmp_path: Path) -> None:
    feedback = _feedback()
    snapshot = resolve_human_holdout(feedback, _config(tmp_path))
    changed = feedback.copy()
    frozen_key = snapshot.validation.iloc[0]
    mask = changed["clip_id"].eq(frozen_key["clip_id"]) & changed["record_time"].eq(
        frozen_key["record_time"]
    )
    changed.loc[mask, "traffic_state"] = (int(frozen_key["traffic_state"]) + 1) % 3

    with pytest.raises(ValueError, match="contradict the frozen holdout"):
        resolve_human_holdout(changed, _config(tmp_path))


def test_creation_requires_three_groups_per_state(tmp_path: Path) -> None:
    limited = _feedback(groups=6)
    with pytest.raises(ValueError, match="three independent groups"):
        resolve_human_holdout(limited, _config(tmp_path))


def test_rejects_accident_and_unvalidated_records(tmp_path: Path) -> None:
    accident = _feedback()
    accident.loc[0, "traffic_state"] = 3
    with pytest.raises(ValueError, match="stable states"):
        resolve_human_holdout(accident, _config(tmp_path))

    unvalidated = _feedback()
    unvalidated.loc[0, "is_human_validated"] = False
    with pytest.raises(ValueError, match="must be human validated"):
        resolve_human_holdout(unvalidated, _config(tmp_path))


def test_rejects_corrupt_snapshot_and_unsafe_pointer(tmp_path: Path) -> None:
    snapshot = resolve_human_holdout(_feedback(), _config(tmp_path))
    with zipfile.ZipFile(snapshot.path, "a") as archive:
        archive.writestr("unexpected/record.csv", "unsafe")
    with pytest.raises(ValueError, match="unexpected or unsafe"):
        FileSystemHoldoutStore(tmp_path).load_snapshot(snapshot.path)

    pointer = json.loads((tmp_path / CURRENT_POINTER_FILE).read_text(encoding="utf-8"))
    pointer["filename"] = "../outside.zip"
    (tmp_path / CURRENT_POINTER_FILE).write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        FileSystemHoldoutStore(tmp_path).load_current()


def test_holdout_comparison_requires_same_fingerprint(tmp_path: Path) -> None:
    first = resolve_human_holdout(_feedback(), _config(tmp_path / "first"))
    second = resolve_human_holdout(_feedback(start=30), _config(tmp_path / "second"))

    require_comparable_holdouts(first.descriptor, first.descriptor)
    with pytest.raises(ValueError, match="different human holdout"):
        require_comparable_holdouts(first.descriptor, second.descriptor)
