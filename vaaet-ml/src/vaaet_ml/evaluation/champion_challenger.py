# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only Champion--Challenger evaluation for VAAET model bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from vaaet.artifacts import validate_manifest
from vaaet.calibration import apply_temperature_scaling, multiclass_brier_score
from vaaet.inference.traffic_state import classify_telemetry_dataframe

from vaaet_ml.evaluation.reporting import (
    build_classification_support_table,
    expected_calibration_error,
    expected_confusion_cost,
)
from vaaet_ml.settings import FEATURE_COLS, MODEL_STATE_LABELS, RANDOM_SEED, SCALER_PATH
from vaaet_ml.training.holdout import HumanHoldoutSnapshot, require_comparable_holdouts
from vaaet_ml.training.lifecycle import ModelInputPolicy, apply_model_input_policy

__all__ = [
    "ChampionChallengerComparison",
    "EvaluationBundle",
    "ModelEvaluation",
    "evaluate_champion_challenger",
    "load_evaluation_bundle",
    "paired_bootstrap_intervals",
    "plot_champion_challenger_confusion",
    "validate_evaluation_pair",
]


class PredictionModel(Protocol):
    """Minimal Keras-compatible boundary used by the evaluator."""

    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray:
        """Return one Normal/Reduced/Congested probability vector per record."""


class FeatureScaler(Protocol):
    """Minimal fitted scaler boundary used by the evaluator."""

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform canonical feature rows without fitting."""


@dataclass(frozen=True)
class EvaluationBundle:
    """A validated serving bundle loaded only for offline evaluation."""

    name: str
    path: Path
    manifest: Mapping[str, object]
    model: PredictionModel
    scaler: FeatureScaler


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics and predictions for one bundle on the immutable test partition."""

    name: str
    metrics: Mapping[str, float]
    support_table: pd.DataFrame
    classified: pd.DataFrame
    probabilities: np.ndarray


@dataclass(frozen=True)
class ChampionChallengerComparison:
    """Paired offline evidence; it deliberately contains no promotion decision."""

    champion: ModelEvaluation
    challenger: ModelEvaluation
    summary: pd.DataFrame
    bootstrap_intervals: pd.DataFrame


def _mapping_value(manifest: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Bundle manifest has an invalid {key!r} block.")
    return cast(Mapping[str, object], value)


def _holdout_descriptor(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = manifest.get("human_holdout")
    if not isinstance(value, Mapping):
        raise ValueError("Champion and challenger require a frozen human holdout descriptor.")
    return cast(Mapping[str, object], value)


def _bundle_input_policy(bundle: EvaluationBundle) -> ModelInputPolicy:
    lifecycle = _mapping_value(bundle.manifest, "training_lifecycle")
    try:
        return ModelInputPolicy(str(lifecycle["input_policy"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Bundle {bundle.name!r} has an invalid input policy.") from exc


def _bundle_decision_policy(bundle: EvaluationBundle) -> Mapping[str, object]:
    return _mapping_value(bundle.manifest, "decision_policy")


def _bundle_model_version(bundle: EvaluationBundle) -> str:
    version = bundle.manifest.get("model_version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Bundle {bundle.name!r} has no model version.")
    return version


def _default_model_loader(path: Path) -> PredictionModel:
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on optional training extra
        raise RuntimeError(
            "Champion--Challenger evaluation requires the VAAET training extra."
        ) from exc
    return cast(PredictionModel, keras.models.load_model(path))


def _default_scaler_loader(path: Path) -> FeatureScaler:
    return cast(FeatureScaler, joblib.load(path))


def load_evaluation_bundle(
    bundle_dir: str | Path,
    *,
    name: str,
    model_loader: Callable[[Path], PredictionModel] | None = None,
    scaler_loader: Callable[[Path], FeatureScaler] | None = None,
) -> EvaluationBundle:
    """Validate a portable serving bundle before loading its binary artifacts."""
    directory = Path(bundle_dir).resolve()
    manifest = validate_manifest(directory)
    load_model = model_loader or _default_model_loader
    load_scaler = scaler_loader or _default_scaler_loader
    return EvaluationBundle(
        name=name,
        path=directory,
        manifest=manifest,
        model=load_model(directory / "traffic_classifier.keras"),
        scaler=load_scaler(directory / Path(SCALER_PATH).name),
    )


def _validate_holdout_records(holdout: HumanHoldoutSnapshot) -> pd.DataFrame:
    test = holdout.test.copy().sort_values(["clip_id", "record_time"]).reset_index(drop=True)
    if test.empty:
        raise ValueError("Frozen human holdout test partition is empty.")
    if "is_human_validated" not in test or not test["is_human_validated"].fillna(False).all():
        raise ValueError("Frozen holdout test records must all be human validated.")
    if "data_origin" in test and test["data_origin"].eq("synthetic").any():
        raise ValueError("Synthetic records are forbidden in the frozen human holdout.")
    if "traffic_state" not in test:
        raise ValueError("Frozen holdout test records must declare traffic_state.")
    labels = pd.to_numeric(test["traffic_state"], errors="coerce")
    if labels.isna().any() or not labels.astype(int).isin(MODEL_STATE_LABELS).all():
        raise ValueError("Frozen holdout must contain only the three stable traffic states.")
    missing_features = [column for column in FEATURE_COLS if column not in test]
    if missing_features:
        raise ValueError(f"Frozen holdout misses canonical features: {missing_features}")
    return test


def validate_evaluation_pair(
    champion: EvaluationBundle,
    challenger: EvaluationBundle,
    holdout: HumanHoldoutSnapshot,
) -> None:
    """Prove both bundles target the exact frozen benchmark before scoring."""
    champion_holdout = _holdout_descriptor(champion.manifest)
    challenger_holdout = _holdout_descriptor(challenger.manifest)
    require_comparable_holdouts(champion_holdout, challenger_holdout)

    expected_fingerprint = champion_holdout.get("fingerprint")
    actual_fingerprint = holdout.descriptor.get("fingerprint")
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Configured holdout snapshot does not match the fingerprint declared by both bundles."
        )
    _validate_holdout_records(holdout)


def _calibrated_probabilities(bundle: EvaluationBundle, frame: pd.DataFrame) -> np.ndarray:
    input_policy = _bundle_input_policy(bundle)
    matrix = apply_model_input_policy(frame, input_policy)
    values = np.asarray(bundle.scaler.transform(matrix.to_numpy()), dtype=float)
    raw_probabilities = np.asarray(bundle.model.predict(values, verbose=0), dtype=float)
    expected_shape = (len(frame), len(MODEL_STATE_LABELS))
    if raw_probabilities.shape != expected_shape:
        raise ValueError(
            f"Bundle {bundle.name!r} returned probabilities with shape {raw_probabilities.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(raw_probabilities).all() or (raw_probabilities < 0).any():
        raise ValueError(f"Bundle {bundle.name!r} returned invalid probabilities.")
    policy = _bundle_decision_policy(bundle)
    temperature = policy.get("temperature", 1.0)
    return apply_temperature_scaling(raw_probabilities, float(temperature))


def _evaluate_bundle(bundle: EvaluationBundle, test: pd.DataFrame) -> ModelEvaluation:
    probabilities = _calibrated_probabilities(bundle, test)
    classified = classify_telemetry_dataframe(
        test,
        bundle.model,
        bundle.scaler,
        model_version=_bundle_model_version(bundle),
        decision_policy=_bundle_decision_policy(bundle),
        input_policy=_bundle_input_policy(bundle),
    )
    truth = test["traffic_state"].to_numpy(dtype=int)
    final_predictions = classified["traffic_state"].to_numpy(dtype=int)
    direct_predictions = classified["model_traffic_state"].to_numpy(dtype=int)
    metrics = {
        "f1_macro": float(
            f1_score(truth, final_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "expected_confusion_cost": expected_confusion_cost(truth, final_predictions),
        "ece": expected_calibration_error(truth, probabilities),
        "brier_score": multiclass_brier_score(truth, probabilities),
        "normal_congested_error": float(
            (((truth == 0) & (final_predictions == 2)) | ((truth == 2) & (final_predictions == 0))).mean()
        ),
        "direct_f1_macro": float(
            f1_score(truth, direct_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
    }
    if classified["traffic_state"].eq(3).any():
        raise RuntimeError("Evaluation violated the invariant: automatic Accident was emitted.")
    return ModelEvaluation(
        name=bundle.name,
        metrics=metrics,
        support_table=build_classification_support_table(truth, final_predictions),
        classified=classified,
        probabilities=probabilities,
    )


def _metric_values(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    values = {
        "f1_macro": float(
            f1_score(truth, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        )
    }
    for state, label in MODEL_STATE_LABELS.items():
        support = int((truth == state).sum())
        values[f"recall_{label.lower()}"] = (
            float(((truth == state) & (predictions == state)).sum() / support)
            if support
            else 0.0
        )
    return values


def paired_bootstrap_intervals(
    y_true: np.ndarray | list[int],
    champion_predictions: np.ndarray | list[int],
    challenger_predictions: np.ndarray | list[int],
    *,
    samples: int = 1_000,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return deterministic paired 95% intervals for key evaluation deltas."""
    truth = np.asarray(y_true, dtype=int)
    champion = np.asarray(champion_predictions, dtype=int)
    challenger = np.asarray(challenger_predictions, dtype=int)
    if samples < 1:
        raise ValueError("Bootstrap samples must be at least one.")
    if truth.ndim != 1 or champion.shape != truth.shape or challenger.shape != truth.shape:
        raise ValueError("Bootstrap inputs must be equally sized one-dimensional arrays.")
    if not len(truth):
        raise ValueError("Bootstrap requires at least one evaluated record.")
    if not set(truth).union(champion, challenger).issubset(MODEL_STATE_LABELS):
        raise ValueError("Bootstrap comparison accepts only the three stable traffic states.")

    champion_metrics = _metric_values(truth, champion)
    challenger_metrics = _metric_values(truth, challenger)
    deltas = {name: [] for name in champion_metrics}
    generator = np.random.default_rng(random_state)
    for indices in generator.integers(0, len(truth), size=(samples, len(truth))):
        sample_truth = truth[indices]
        sample_champion = _metric_values(sample_truth, champion[indices])
        sample_challenger = _metric_values(sample_truth, challenger[indices])
        for name in deltas:
            deltas[name].append(sample_challenger[name] - sample_champion[name])

    return pd.DataFrame(
        {
            "metric": list(champion_metrics),
            "champion": [champion_metrics[name] for name in champion_metrics],
            "challenger": [challenger_metrics[name] for name in champion_metrics],
            "delta_challenger_minus_champion": [
                challenger_metrics[name] - champion_metrics[name] for name in champion_metrics
            ],
            "ci_95_low": [float(np.quantile(deltas[name], 0.025)) for name in champion_metrics],
            "ci_95_high": [float(np.quantile(deltas[name], 0.975)) for name in champion_metrics],
        }
    )


def plot_champion_challenger_confusion(
    comparison: ChampionChallengerComparison,
    y_true: np.ndarray | list[int],
) -> None:
    """Render final stable-state confusion matrices side by side for human review."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    truth = np.asarray(y_true, dtype=int)
    champion = comparison.champion.classified["traffic_state"].to_numpy(dtype=int)
    challenger = comparison.challenger.classified["traffic_state"].to_numpy(dtype=int)
    if champion.shape != truth.shape or challenger.shape != truth.shape:
        raise ValueError("Confusion-matrix truth and prediction lengths must match.")
    labels = list(MODEL_STATE_LABELS)
    names = [MODEL_STATE_LABELS[label] for label in labels]
    matrices = (
        (comparison.champion.name, confusion_matrix(truth, champion, labels=labels)),
        (comparison.challenger.name, confusion_matrix(truth, challenger, labels=labels)),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, (name, matrix) in zip(axes, matrices, strict=True):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
            ax=axis,
        )
        axis.set(title=name, xlabel="Predicción final", ylabel="Estado humano")
    figure.tight_layout()
    plt.show()


def evaluate_champion_challenger(
    champion: EvaluationBundle,
    challenger: EvaluationBundle,
    holdout: HumanHoldoutSnapshot,
    *,
    bootstrap_samples: int = 1_000,
    random_state: int = RANDOM_SEED,
) -> ChampionChallengerComparison:
    """Evaluate two compatible bundles without selecting thresholds or promoting either one."""
    validate_evaluation_pair(champion, challenger, holdout)
    test = _validate_holdout_records(holdout)
    champion_result = _evaluate_bundle(champion, test)
    challenger_result = _evaluate_bundle(challenger, test)
    summary = pd.DataFrame(
        {
            "metric": list(champion_result.metrics),
            champion.name: list(champion_result.metrics.values()),
            challenger.name: list(challenger_result.metrics.values()),
        }
    )
    summary["delta_challenger_minus_champion"] = (
        summary[challenger.name] - summary[champion.name]
    )
    bootstrap = paired_bootstrap_intervals(
        test["traffic_state"].to_numpy(dtype=int),
        champion_result.classified["traffic_state"].to_numpy(dtype=int),
        challenger_result.classified["traffic_state"].to_numpy(dtype=int),
        samples=bootstrap_samples,
        random_state=random_state,
    )
    return ChampionChallengerComparison(
        champion=champion_result,
        challenger=challenger_result,
        summary=summary,
        bootstrap_intervals=bootstrap,
    )
