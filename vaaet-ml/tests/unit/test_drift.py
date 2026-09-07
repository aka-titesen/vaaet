# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from vaaet.artifacts import FEATURE_SCHEMA_VERSION

from vaaet_ml.evaluation.drift import (
    build_feature_cohort,
    build_feature_cohort_from_raw_telemetry,
    compare_feature_cohorts,
)
from vaaet_ml.settings import FEATURE_COLS


def _cohort(*, offset: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number in range(6):
        row: dict[str, object] = {
            "clip_id": f"clip-{number}",
            "record_time": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=number),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "data_origin": "real",
        }
        row.update({feature: float(number) + offset for feature in FEATURE_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_drift_reports_all_canonical_features_and_quantifies_changes() -> None:
    report = compare_feature_cohorts(
        build_feature_cohort(_cohort(), name="reference"),
        build_feature_cohort(_cohort(offset=5.0), name="operational"),
    )

    assert report.reference.profile.records == 6
    assert report.operational.profile.provenance == {"real": 6}
    assert set(report.summary["feature"]) == set(FEATURE_COLS)
    assert report.summary["psi"].notna().all()
    assert report.summary["median_delta"].eq(5.0).all()


def test_drift_rejects_incompatible_legacy_schema() -> None:
    legacy = _cohort()
    legacy["feature_schema_version"] = "traffic-features-v1"

    with pytest.raises(ValueError, match="not compatible"):
        build_feature_cohort(legacy, name="legacy")


def test_drift_rejects_empty_duplicate_and_nonfinite_cohorts() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_feature_cohort(_cohort().iloc[0:0], name="empty")

    duplicate = pd.concat([_cohort(), _cohort().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_cohort(duplicate, name="duplicate")

    nonfinite = _cohort()
    nonfinite.loc[0, FEATURE_COLS[0]] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        build_feature_cohort(nonfinite, name="nonfinite")


def test_raw_telemetry_requires_v2_before_feature_engineering() -> None:
    raw = _cohort().drop(columns=["feature_schema_version"])
    raw["telemetry_schema_version"] = "traffic-telemetry-v1"

    with pytest.raises(ValueError, match="traffic-telemetry-v3"):
        build_feature_cohort_from_raw_telemetry(raw, name="legacy-raw")
