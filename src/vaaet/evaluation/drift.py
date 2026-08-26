"""Contract-aware feature-cohort profiling and covariate drift summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.data.timestamps import normalize_timestamp_series
from vaaet.features.engineering import engineer_features
from vaaet.settings import (
    DATA_ORIGIN_COL,
    FEATURE_COLS,
    RANDOM_SEED,
    TELEMETRY_SCHEMA_VERSION,
)

__all__ = [
    "DriftReport",
    "FeatureCohort",
    "FeatureCohortProfile",
    "build_feature_cohort",
    "build_feature_cohort_from_raw_telemetry",
    "compare_feature_cohorts",
    "plot_feature_drift",
]


@dataclass(frozen=True)
class FeatureCohortProfile:
    """Traceable, read-only quality summary for a canonical feature cohort."""

    name: str
    records: int
    clips: int
    time_start: pd.Timestamp
    time_end: pd.Timestamp
    feature_schema_version: str
    provenance: Mapping[str, int]
    summary: pd.DataFrame


@dataclass(frozen=True)
class FeatureCohort:
    """Validated cohort plus its profile; the frame is an analysis copy only."""

    frame: pd.DataFrame
    profile: FeatureCohortProfile


@dataclass(frozen=True)
class DriftReport:
    """Distribution evidence, not an automatic drift or retraining decision."""

    reference: FeatureCohort
    operational: FeatureCohort
    summary: pd.DataFrame


def _require_feature_schema(frame: pd.DataFrame) -> str:
    if "feature_schema_version" not in frame:
        raise ValueError("Feature cohort must declare feature_schema_version.")
    versions = set(frame["feature_schema_version"].dropna().astype(str))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(
            "Feature cohort is not compatible with traffic-features-v2: "
            f"{sorted(versions) or ['missing']}"
        )
    return FEATURE_SCHEMA_VERSION


def _numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.loc[:, FEATURE_COLS]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any().any():
        columns = invalid.columns[invalid.any()].tolist()
        raise ValueError(f"Feature cohort has non-numeric values: {columns}")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Feature cohort contains non-finite feature values.")
    return numeric


def _feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.loc[:, FEATURE_COLS]
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLS:
        values = numeric[feature].dropna()
        rows.append(
            {
                "feature": feature,
                "records": int(len(numeric)),
                "missing": int(numeric[feature].isna().sum()),
                "missing_rate": float(numeric[feature].isna().mean()),
                "p05": float(values.quantile(0.05)) if not values.empty else float("nan"),
                "p25": float(values.quantile(0.25)) if not values.empty else float("nan"),
                "median": float(values.median()) if not values.empty else float("nan"),
                "p75": float(values.quantile(0.75)) if not values.empty else float("nan"),
                "p95": float(values.quantile(0.95)) if not values.empty else float("nan"),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25))
                if not values.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_feature_cohort(frame: pd.DataFrame, *, name: str) -> FeatureCohort:
    """Validate and profile an immutable, canonical 19-feature analysis cohort."""
    if not name.strip():
        raise ValueError("Feature cohort name must be non-empty.")
    required = {"clip_id", "record_time", "feature_schema_version", *FEATURE_COLS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature cohort misses required columns: {missing}")
    if frame.empty:
        raise ValueError("Feature cohort must not be empty.")

    prepared = frame.copy()
    prepared["record_time"] = normalize_timestamp_series(prepared["record_time"])
    if prepared.duplicated(["clip_id", "record_time"]).any():
        raise ValueError("Feature cohort contains duplicate clip/time records.")
    schema = _require_feature_schema(prepared)
    prepared.loc[:, FEATURE_COLS] = _numeric_feature_frame(prepared)
    origins = (
        prepared.get(DATA_ORIGIN_COL, pd.Series("unknown", index=prepared.index))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    profile = FeatureCohortProfile(
        name=name,
        records=len(prepared),
        clips=int(prepared["clip_id"].astype(str).nunique()),
        time_start=prepared["record_time"].min(),
        time_end=prepared["record_time"].max(),
        feature_schema_version=schema,
        provenance={str(key): int(value) for key, value in origins.items()},
        summary=_feature_summary(prepared),
    )
    return FeatureCohort(frame=prepared, profile=profile)


def build_feature_cohort_from_raw_telemetry(
    telemetry: pd.DataFrame, *, name: str
) -> FeatureCohort:
    """Engineer a v2 raw-telemetry snapshot before profiling its 19 features."""
    if "telemetry_schema_version" not in telemetry:
        raise ValueError("Raw telemetry cohort must declare telemetry_schema_version.")
    versions = set(telemetry["telemetry_schema_version"].dropna().astype(str))
    if versions != {TELEMETRY_SCHEMA_VERSION}:
        raise ValueError(
            "Raw telemetry cohort is not compatible with traffic-telemetry-v2: "
            f"{sorted(versions) or ['missing']}"
        )
    features = engineer_features(telemetry).copy()
    features["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    return build_feature_cohort(features, name=name)


def _population_stability_index(
    reference: pd.Series, operational: pd.Series, *, bins: int
) -> float:
    baseline = reference.dropna().to_numpy(dtype=float)
    observed = operational.dropna().to_numpy(dtype=float)
    if not len(baseline) or not len(observed):
        return float("nan")
    if bins < 2:
        raise ValueError("PSI requires at least two bins.")

    quantile_edges = np.quantile(baseline, np.linspace(0.0, 1.0, bins + 1))
    inner_edges = np.unique(quantile_edges[1:-1])
    if not len(inner_edges):
        centre = float(baseline[0])
        epsilon = max(abs(centre) * 1e-6, 1e-6)
        inner_edges = np.array([centre - epsilon, centre + epsilon])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    baseline_counts, _ = np.histogram(baseline, bins=edges)
    observed_counts, _ = np.histogram(observed, bins=edges)
    smoothing = 1e-6
    baseline_share = (baseline_counts + smoothing) / (baseline_counts.sum() + smoothing * len(baseline_counts))
    observed_share = (observed_counts + smoothing) / (observed_counts.sum() + smoothing * len(observed_counts))
    return float(np.sum((observed_share - baseline_share) * np.log(observed_share / baseline_share)))


def compare_feature_cohorts(
    reference: FeatureCohort,
    operational: FeatureCohort,
    *,
    psi_bins: int = 10,
) -> DriftReport:
    """Quantify covariate changes without deriving a retraining threshold or action."""
    if reference.profile.feature_schema_version != operational.profile.feature_schema_version:
        raise ValueError("Feature cohorts use different feature schema versions.")
    reference_summary = reference.profile.summary.set_index("feature")
    operational_summary = operational.profile.summary.set_index("feature")
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLS:
        baseline = reference_summary.loc[feature]
        observed = operational_summary.loc[feature]
        rows.append(
            {
                "feature": feature,
                "reference_median": baseline["median"],
                "operational_median": observed["median"],
                "median_delta": observed["median"] - baseline["median"],
                "reference_iqr": baseline["iqr"],
                "operational_iqr": observed["iqr"],
                "iqr_delta": observed["iqr"] - baseline["iqr"],
                "reference_missing_rate": baseline["missing_rate"],
                "operational_missing_rate": observed["missing_rate"],
                "missing_rate_delta": observed["missing_rate"] - baseline["missing_rate"],
                "psi": _population_stability_index(
                    reference.frame[feature], operational.frame[feature], bins=psi_bins
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    return DriftReport(reference=reference, operational=operational, summary=summary.reset_index(drop=True))


def plot_feature_drift(
    report: DriftReport,
    *,
    features: Sequence[str] | None = None,
    max_features: int = 6,
    sample_size: int = 20_000,
    random_state: int = RANDOM_SEED,
) -> None:
    """Plot bounded, deterministic distribution overlays for selected canonical features."""
    import matplotlib.pyplot as plt

    if max_features < 1 or sample_size < 1:
        raise ValueError("max_features and sample_size must be positive.")
    selected = list(features) if features is not None else report.summary["feature"].head(max_features).tolist()
    unknown = sorted(set(selected) - set(FEATURE_COLS))
    if unknown:
        raise ValueError(f"Unknown canonical features requested for drift plot: {unknown}")
    selected = selected[:max_features]
    if not selected:
        raise ValueError("At least one feature is required for drift plotting.")

    rows = int(np.ceil(len(selected) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), squeeze=False)
    for axis, feature in zip(axes.ravel(), selected):
        reference = report.reference.frame[feature].dropna()
        operational = report.operational.frame[feature].dropna()
        if len(reference) > sample_size:
            reference = reference.sample(sample_size, random_state=random_state)
        if len(operational) > sample_size:
            operational = operational.sample(sample_size, random_state=random_state)
        axis.hist(reference, bins=30, density=True, alpha=0.55, label=report.reference.profile.name)
        axis.hist(operational, bins=30, density=True, alpha=0.55, label=report.operational.profile.name)
        psi = float(report.summary.set_index("feature").loc[feature, "psi"])
        axis.set(title=f"{feature} | PSI={psi:.4f}", xlabel=feature, ylabel="Densidad")
        axis.legend()
        axis.grid(True, alpha=0.2)
    for axis in axes.ravel()[len(selected) :]:
        axis.set_visible(False)
    figure.tight_layout()
    plt.show()
